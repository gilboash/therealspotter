"""
The race day's running order on fril.co.il/live.

The same way FPVTrackside sends fril its live events — an HTTP PUT of one
JSON object to fril's ingest address, with its token
(https://fril.co.il/api/live/ingest?token=…) — the app sends the race on now
and every race to come (race_day/schedule.py), as one UpcomingRaces event:

  {"type": "UpcomingRaces", "ts": "…Z", "source": "lazy-spotter", "event": "…",
   "current": <race or null>,
   "races": [<race>, …],             the races to come, next first
   "start": {"at": "…Z", "paused", "round", "race"} or null,
                                     FPVTrackside's Auto Runner counting down to that
                                     race's start (GET /api/v1/app), or null
   "pbLaps": n or null}              the event's consecutive laps (Event.json PBLaps):
                                     fril ranks its qualifying board by it
  race = {"label", "stage", "round", "race", "raceType",
          "pilots": [{"name", "channel": {"label", "band", "number",
                                          "frequency", "colorR", "colorG", "colorB"}}]}

It is sent whenever that changes (a race starts or ends, a race or a line-up
changes in FPVTrackside) and again every RESEND_S, because fril keeps its live
board in memory only and shows the list only while it is fresh.

Once the brackets are made, the finals' mains go the same way, as one
Brackets event (brackets_payload): every main as FPVTrackside's Seeder has it
(GET /api/v1/stages/{id}: its diagram, races and standings), with the race on
now and the next one marked, so fril.co.il/live can draw them like the
Seeder's page. fril keeps them until its board is reset; an empty list takes
them back off.

  {"type": "Brackets", "ts": "…Z", "source": "lazy-spotter", "event": "…",
   "stages": [{"name", "className", "main", "format", "formatName", "layout",
               "isComplete", "champion", "raceCount", "racesDone",
               "races": [{"id", "round", "group", "title", "state", "live", "pilots", "winners"}],
               "diagrams": [{"title", "nodes": [{"id", "column", "lane", "label",
                             "seats": [{"name", "position", "advanced", "fromRace", "fromRank"}]}],
                             "edges": [{"from", "to", "ranks"}]}],
               "standings": [{"rank", "name", "detail", "tied"}]}]} It carries
no "seq": that is FPVTrackside's, and fril uses it to notice FPVTrackside
restarting.

When a finished heat's laps change in FPVTrackside (the app's correction, or a
marshal's edit there), the heat's laps as they are now go as one HeatLaps
event (heat_laps_payload) — FPVTrackside never resends a past heat's laps:

  {"type": "HeatLaps", "ts", "source", "event", "round", "race",
   "pilots": [{"name", "position" (or null), "laps": [{"lapNumber", "sectorTime"}]}]}
                                     lapNumber 0 = the holeshot (its time from the start) Sending runs on its own thread: a slow or unreachable fril never
holds up the app.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

RESEND_S = 60.0          # fril shows the list while it's fresh (3 min): re-sent this often
RETRY_S = 10.0           # after a failed send, the next try
TIMEOUT = 5.0
# fril.co.il is behind Cloudflare, which refuses Python's own "Python-urllib/…" (HTTP 403)
USER_AGENT = "LazySpotter/2 (race-day app)"
MAX_RACES = 30           # what fril keeps


def _rgb(hexcolor: str):
    h = (hexcolor or "").lstrip("#")
    if len(h) != 6:
        return None, None, None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None, None, None


def _race(r: dict) -> dict:
    pilots = []
    for p in r["pilots"]:
        red, green, blue = _rgb(p.get("color"))
        pilots.append({"name": p["name"], "channel": {
            "label": p.get("channel"), "band": p.get("band"), "number": p.get("number"),
            "frequency": p.get("frequency"), "colorR": red, "colorG": green, "colorB": blue}})
    return {"label": r["label"], "stage": r.get("stage") or None, "round": r.get("round"), "race": r.get("race"),
            "raceType": r.get("type"), "pilots": pilots}


def payload(schedule: dict) -> dict:
    """The UpcomingRaces event for a schedule (race_day/schedule.py)."""
    cur = schedule.get("current")
    st = schedule.get("start")                   # FPVTrackside's Auto Runner, counting down to a race's start
    start = {"at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(st["at"])) + f".{int(st['at'] % 1 * 1000):03d}Z",
             "paused": bool(st.get("paused")), "round": st.get("round"), "race": st.get("race")} if st else None
    return {"type": "UpcomingRaces", "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source": "lazy-spotter", "event": schedule.get("event"),
            "current": _race(cur) if cur else None,
            "races": [_race(r) for r in schedule.get("upcoming", [])[:MAX_RACES]],
            "start": start, "pbLaps": schedule.get("pb_laps")}


def heat_laps_payload(event: str, rnd: int, race: int, pilots: list) -> dict:
    """The HeatLaps event of one heat. pilots: [{"name", "position" (or None), "times": the valid crossings'
    clock times, holeshot first when there is one, "holeshot": bool, "start": the race's start}]."""
    out = []
    for p in pilots:
        ts = sorted(p["times"])
        laps = []
        if ts and p.get("holeshot"):
            laps.append({"lapNumber": 0, "sectorTime": round(ts[0] - p["start"], 3)})
            prev, rest = ts[0], ts[1:]
        else:
            prev, rest = p["start"], ts
        for i, t in enumerate(rest, 1):
            laps.append({"lapNumber": i, "sectorTime": round(t - prev, 3)})
            prev = t
        out.append({"name": p["name"], "position": p.get("position"), "laps": laps})
    return {"type": "HeatLaps", "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source": "lazy-spotter", "event": event, "round": rnd, "race": race, "pilots": out}


def _names(lst) -> list:
    return [p if isinstance(p, str) else (p or {}).get("name") for p in (lst or [])
            if (p if isinstance(p, str) else (p or {}).get("name"))]


def _stage(cls: str, main: str, st: dict, now_id: str, next_id: str) -> dict:
    def live(r):
        fid = (r.get("fpvRaceId") or "").lower()
        return "now" if fid and fid == now_id else "next" if fid and fid == next_id else None
    races = [{"id": r.get("id"), "round": r.get("round"), "group": r.get("group"), "title": r.get("title"),
              "state": r.get("state"), "live": live(r), "pilots": _names(r.get("pilots")),
              "winners": _names(r.get("winners"))} for r in st.get("races") or []]
    diagrams = [{"title": d.get("title"),
                 "nodes": [{"id": n.get("id"), "column": n.get("column"), "lane": n.get("lane"),
                            "label": n.get("label"),
                            "seats": [{"name": x.get("name"), "position": x.get("position"),
                                       "advanced": bool(x.get("advanced")), "fromRace": x.get("fromRace"),
                                       "fromRank": x.get("fromRank")} for x in n.get("seats") or []]}
                           for n in d.get("nodes") or []],
                 "edges": [{"from": e.get("from"), "to": e.get("to"), "ranks": e.get("ranks") or []}
                           for e in d.get("edges") or []]} for d in st.get("diagrams") or []]
    return {"name": st.get("name") or f"{cls} {main} main", "className": cls, "main": main,
            "format": st.get("format"), "formatName": st.get("formatName"), "layout": st.get("layout"),
            "isComplete": bool(st.get("isComplete")), "champion": st.get("champion"),
            "raceCount": st.get("raceCount"), "racesDone": st.get("racesDone"),
            "races": races, "diagrams": diagrams,
            "standings": [{"rank": x.get("rank"), "name": x.get("name"), "detail": x.get("detail"),
                           "tied": bool(x.get("tied"))} for x in st.get("standings") or [] if x.get("name")]}


def brackets_payload(mains: list, schedule: Optional[dict]) -> dict:
    """The Brackets event: mains = [(class, main, stage dict from GET /stages/{id})], in running order.
    The schedule (race_day/schedule.py) says which FPVTrackside race is on now and which is next."""
    schedule = schedule or {}
    now_id = ((schedule.get("current") or {}).get("id") or "").lower()
    nxt = (schedule.get("upcoming") or [None])[0] or {}
    next_id = (nxt.get("id") or "").lower()
    return {"type": "Brackets", "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source": "lazy-spotter", "event": schedule.get("event"),
            "stages": [_stage(c, m, st, now_id, next_id) for c, m, st in mains if st]}


def _key(p: dict) -> str:
    """What fril shows: everything but the send time."""
    return json.dumps({k: v for k, v in p.items() if k != "ts"}, sort_keys=True)


class FrilLive:
    def __init__(self, url: str, log=print):
        self.url, self.log = url.strip(), log
        self.last_ok: Optional[float] = None
        self.error: Optional[str] = None
        self._key, self._sent_at = None, 0.0
        self._bkey, self._bsent_at = None, 0.0        # the brackets: what was last sent, when
        self._pending: dict = {}                      # type → the newest event of that type to send
        self._lock = threading.Lock()
        self._wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def update(self, schedule: Optional[dict]):
        """Called with the current schedule (often): sends when it changed, or
        when the last send is RESEND_S old."""
        if not schedule or not schedule.get("has"):
            return
        p = payload(schedule)
        key = _key(p)
        if key == self._key and time.time() - self._sent_at < RESEND_S:
            return
        self._key, self._sent_at = key, time.time()
        self._put(p)

    def update_brackets(self, p: Optional[dict]):
        """Called with the Brackets event (brackets_payload) after each look at the Seeder: sends when it
        changed, or when the last send is RESEND_S old. An empty list is sent once, to take them off fril."""
        if p is None:
            return
        key = _key(p)
        if not p["stages"] and self._bkey is None:
            return                               # never sent any: nothing to take back
        if key == self._bkey and time.time() - self._bsent_at < RESEND_S:
            return
        if not p["stages"] and key == self._bkey:
            return                               # taken back already
        self._bkey, self._bsent_at = key, time.time()
        self._put(p)

    def update_heat_laps(self, p: dict):
        """A finished heat's laps after a change (heat_laps_payload): sent once, each heat on its own."""
        if p.get("round") is None or p.get("race") is None:
            return
        self._put(p)

    def _put(self, p: dict):
        with self._lock:                         # only the newest of each kind (and of each heat's laps) matters
            k = f"HeatLaps:{p['round']}:{p['race']}" if p["type"] == "HeatLaps" else p["type"]
            self._pending[k] = p
        self._wake.set()

    def _worker(self):
        while True:
            self._wake.wait()
            with self._lock:
                todo, self._pending = list(self._pending.values()), {}
                self._wake.clear()
            for p in todo:
                self._send(p)

    def _send(self, p: dict):
        req = urllib.request.Request(self.url, data=json.dumps(p).encode(), method="PUT",
                                     headers={"Content-Type": "application/json; charset=utf-8",
                                              "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                r.read()
            if self.error:
                self.log("fril live: sending again")
            self.last_ok, self.error = time.time(), None
        except urllib.error.HTTPError as e:
            self._failed(p, "the token was refused (HTTP 401)" if e.code == 401 else
                         "blocked by fril's firewall (HTTP 403)" if e.code == 403 else f"HTTP {e.code}")
        except (urllib.error.URLError, OSError) as e:
            self._failed(p, str(getattr(e, "reason", e)))

    def _failed(self, p: dict, why: str):
        if why != self.error:
            what = {"Brackets": "the brackets", "HeatLaps": "a heat's corrected laps"}.get(p["type"], "the next races")
            self.log(f"fril live: couldn't send {what} — {why}")
        self.error = why
        if p["type"] == "Brackets":       # sent again at the next look at the Seeder (an empty list too)
            self._bkey = ""
        elif p["type"] == "HeatLaps":     # tried again in RETRY_S
            threading.Timer(RETRY_S, self._put, args=(p,)).start()
        else:                             # try again in RETRY_S
            self._sent_at = time.time() - RESEND_S + RETRY_S

    def status(self) -> dict:
        return {"url": self.url, "last_ok": self.last_ok, "error": self.error}
