"""
Read-only dashboard server for race day.

Serves every event of the data folder — tracks and FPVTrackside events, each
with its data in <event>/lazySpotter/ — to any browser on the same Wi-Fi.
It starts with the app and follows the data folder. The event open in the
app is read live (with its status); the others from their saved files. Only
GET requests: nothing can be changed from the dashboard. Videos are streamed
with HTTP range requests, which phone browsers need to play and seek.

    /                       the dashboard (static/)
    /api/events             every event: name, kind, date, runs, learned, open in the app
  each of these takes the event as ?e=<id> (its folder relative to the data folder):
    /api/status             mode, phase, learn progress, live channels, queue
    /api/runs               all runs with their summary and speed index
    /api/results            the event's results: per class (the pilots of one track layout) the leaderboard
                            rows, the field, the named sections; and every heat with its finishing order
    /api/ghost?p=A&p=B      two pilots' best laps of the event, to play side by side
    /api/run/<stem>         one run: stats, passes, laps, comparison with the field
    /api/heat/<n>           the runs of a heat, with passes for the gap chart
    /api/pilot/<name>       a pilot's runs and section strengths
    /api/track              learned gates (images), sections, hardest sections
    /api/schedule           FPVTrackside events: every race in running order, the current race, the races
                            to come, and each pilot's next race (race_day/schedule.py)
    /api/race/heats         the race view (static/race/): every heat it can play (race_day/raceview.py)
    /api/race/heat/<n>      a heat for the race view: its video, each pilot's passes, laps and stats, the track
    /video/<stem>.mp4       a run's video
    /img?p=<path>           an image inside the event's data folder (gate crops)
"""

import json
import mimetypes
import os
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dataset_paths import TrackPaths
from race_day import raceview
from race_day import stats as rstats
from race_day.eventsetup import LAYOUTS_DIR, MAIN, _slug, section_legs

STATIC = Path(__file__).parent / "static"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class DataView:
    """Everything the dashboard shows, read from the track folder (plus the
    controller's live status). Stats files are cached by modification time."""

    def __init__(self, day, event_id: str = ""):
        self.day = day
        self.q = "?e=" + urllib.parse.quote(event_id) if event_id else ""
        self._cache = {}
        self._lock = threading.Lock()
        self._schedule = None

    @property
    def tp(self):
        return self.day.tp

    def setup(self) -> dict:
        """The event's setup (race_day/eventsetup.py): its classes, layouts and named sections."""
        return self._json(self.tp.dir / "event_setup.json") or {}

    def layouts(self) -> Dict[str, TrackPaths]:
        """Every track layout of the event → its track folder; main (the event's own) first."""
        out = {MAIN: self.tp}
        for name in (self.setup().get("layouts") or {}):
            if name != MAIN:
                out[name] = TrackPaths(self.tp.dir / LAYOUTS_DIR / _slug(name))
        return out

    def _tpx(self, stem: str):
        """A run's track folder: its class's layout (race_day/eventsetup.py). An event that isn't open in
        the app: the layout that holds the run's stats."""
        f = getattr(self.day, "tp_of_stem", None)
        if f:
            return f(stem)
        for tp in self.layouts().values():
            if (tp.runs_dir / f"{stem}.stats.json").exists():
                return tp
        return self.day.tp

    def layout_of(self, stem: str) -> str:
        d = Path(self._tpx(stem).dir).resolve()
        return next((name for name, tp in self.layouts().items() if Path(tp.dir).resolve() == d), MAIN)

    def pb_laps(self) -> int:
        """The event's laps in a row (FPVTrackside's PB laps: what its qualifying is ranked by); 3 without."""
        n = (getattr(getattr(self.day, "event", None), "rules", None) or {}).get("pb_laps")
        if not isinstance(n, int):
            ev = self._json(self.tp.dir.parent / "Event.json")
            n = ev.get("PBLaps") if isinstance(ev, dict) else None
        return n if isinstance(n, int) and n >= 1 else 3

    def named(self, layout: str, n: int) -> List[dict]:
        """A layout's named sections (Event setup → Sections) with their legs: [{name, from, to, legs}]."""
        out = []
        for x in ((self.setup().get("layouts") or {}).get(layout) or {}).get("sections") or []:
            try:
                legs = section_legs(int(x["from"]), int(x["to"]), n)
            except (KeyError, TypeError, ValueError):
                continue
            if legs and x.get("name"):
                out.append({"name": x["name"], "from": int(x["from"]), "to": int(x["to"]), "legs": legs})
        return out

    def _json(self, path: Path) -> Optional[dict]:
        try:
            mt = path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            hit = self._cache.get(path)
            if hit and hit[0] == mt:
                return hit[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._cache[path] = (mt, data)
        return data

    def visible_runs(self):
        """Every run except void ones (an earlier attempt of a restarted heat):
        those are kept on disk but never shown."""
        with self.day.lock:
            return [dict(r) for r in self.day.state["runs"].values() if not r.get("void")]

    def visible(self, stem: str) -> bool:
        with self.day.lock:
            r = self.day.state["runs"].get(stem)
            return bool(r) and not r.get("void")

    def all_stats(self):
        runs = self.visible_runs()
        out = []
        for r in runs:
            if r["status"] == "labelled":
                st = self._json(self._tpx(r['stem']).runs_dir / f"{r['stem']}.stats.json")
                if st:
                    out.append(dict(st, layout=self.layout_of(r["stem"])))
        return out

    @staticmethod
    def _on(stats: List[dict], layout: str) -> List[dict]:
        """The runs flown on one layout (times on different layouts don't compare), with its gate count."""
        mine = [s for s in stats if s["layout"] == layout]
        counts = [s["n_gates"] for s in mine]
        n = max(set(counts), key=counts.count) if counts else None
        return [s for s in mine if s["n_gates"] == n]

    def field(self, layout: str = MAIN, stats: Optional[List[dict]] = None):
        return rstats.field_stats(self._on(self.all_stats() if stats is None else stats, layout))

    def runs(self):
        every = self.all_stats()
        fields = {lay: self.field(lay, every) for lay in self.layouts()}
        stats = {s["stem"]: s for s in every}
        runs = self.visible_runs()
        out = []
        for r in sorted(runs, key=lambda r: r.get("recorded_at", 0), reverse=True):
            st = stats.get(r["stem"])
            field = fields.get(st["layout"]) if st else None
            if st and field and field["n_gates"] == st["n_gates"]:
                cmp = rstats.compare(st, field)
                r.update(laps=st["laps"], best_lap=st["best_lap"], best3=st["best3"],
                         theoretical_best=st["theoretical_best"], lap_cv=st["lap_cv"],
                         speed_index=cmp["speed_index"], missed=len(st["missed_gates"]),
                         ended_mid_lap=st["ended_mid_lap"], holeshot=st["holeshot"])
            out.append(r)
        return out

    def results(self):
        """The event's results: per class (one layout's pilots) the field, the named sections and one row
        per pilot; then every heat."""
        stats = self.all_stats()
        pb = self.pb_laps()
        heats = self.heats(stats)
        craft = rstats.race_craft(heats)
        classes = []
        for lay in self.layouts():
            mine = self._on(stats, lay)
            if not mine:
                continue
            field = rstats.field_stats(mine)
            named = self.named(lay, field["n_gates"])
            names = [c["name"] for c in self.setup().get("classes") or [] if (c.get("layout") or MAIN) == lay]
            by: Dict[str, List[dict]] = {}
            for s in mine:
                by.setdefault(s["pilot"], []).append(s)
            classes.append({"layout": lay, "name": " / ".join(names) or ("" if lay == MAIN else lay),
                            "n_gates": field["n_gates"], "field": field, "named": rstats.named_field(mine, named),
                            "pilots": [dict(rstats.pilot_summary(p, rs, field, pb, named), craft=craft.get(p))
                                       for p, rs in by.items()]})
        return {"pb_laps": pb, "classes": classes, "heats": heats}

    def ghost(self, names: List[str]):
        """Each pilot's best lap of the event, to play side by side from its start: the run, where the lap
        is in its video, and the time at every gate of the lap (seconds from the lap's start)."""
        every = self.all_stats()
        out = []
        for name in names:
            mine = [s for s in every if s["pilot"] == name and s["best_lap"] is not None and s.get("lap_marks")]
            if not mine:
                return None
            s = min(mine, key=lambda s: s["best_lap"])
            i = s["best_lap_index"]
            if i is None or i >= len(s["lap_marks"]):
                return None
            m = s["lap_marks"][i]
            r = next((x for x in self.visible_runs() if x["stem"] == s["stem"]), None)
            if not r:
                return None
            race = self._json(self._tpx(s["stem"]).race_data(s["stem"])) or {}
            gates = [{"gate": p["gate_id"], "t": round(p["t"] - m["t0"], 3)} for p in race.get("passes", [])
                     if p.get("gate_id", -1) >= 1 and m["t0"] + 0.3 < p["t"] < m["t1"] - 0.3]
            # only gates passed about where the field passes them in a lap: a wrongly numbered gate is left out
            med = self.field(s["layout"], every).get("section_median") or []
            if med and all(v is not None for v in med):
                share = {g + 2: sum(med[:g + 1]) / sum(med) for g in range(len(med) - 1)}     # gate → part of the lap
                gates = [x for x in gates if x["gate"] in share
                         and abs(x["t"] / s["best_lap"] - share[x["gate"]]) < 0.15]
            out.append({"pilot": name, "stem": s["stem"], "heat": s.get("heat"), "lap": i + 1, "time": s["best_lap"],
                        "t0": m["t0"], "t1": m["t1"], "crop": r.get("crop"), "n_gates": s["n_gates"],
                        "gates": gates, "video_url": f"/video/{s['stem']}.mp4{self.q}"})
        return {"laps": out}

    def _races(self) -> Dict[str, dict]:
        """FPVTrackside's finished races by the short id the runs' names start with (trackside.Heat.short)."""
        sc = self.schedule()
        return {r["id"][:8]: r for r in sc.get("races") or [] if r["state"] == "ended" and r["pilots"]} \
            if sc.get("has") else {}

    def heats(self, stats: List[dict]) -> List[dict]:
        """Every heat of the event in the order flown: who flew it, in finishing order, with laps and best
        lap. An FPVTrackside event: its finished races (those without an analysed video too) with
        FPVTrackside's places and laps; else the analysed runs."""
        pb = self.pb_laps()
        st = {s["stem"]: s for s in stats}
        races = self._races()
        by: Dict[int, List[dict]] = {}
        for r in self.visible_runs():
            by.setdefault(r["heat"], []).append(r)

        def from_race(race, runs):
            stem = {r["pilot"]: r["stem"] for r in runs}
            return [{"pilot": p["name"], "position": p["position"], "dnf": p["dnf"], "laps": len(p["laps"]),
                     "best_lap": min(p["laps"]) if p["laps"] else None, "total": p["total"],
                     "best_row": rstats.best_in_a_row(p["laps"], pb), "holeshot": p["holeshot"],
                     "lap_times": p["laps"], "stem": stem.get(p["name"])} for p in race["pilots"]]
        out = []
        for n, runs in by.items():
            race = races.pop(runs[0]["stem"].rsplit("_", 1)[0], None)
            if race:
                pilots = from_race(race, runs)
            else:
                pilots = []
                for r in runs:
                    s = st.get(r["stem"]) or {}
                    lt = s.get("lap_times") or []
                    pilots.append({"pilot": r["pilot"], "position": None, "dnf": False, "laps": len(lt),
                                   "best_lap": min(lt) if lt else None, "total": round(sum(lt), 3) if lt else None,
                                   "best_row": rstats.best_in_a_row(lt, pb), "holeshot": s.get("holeshot"),
                                   "lap_times": lt, "stem": r["stem"]})
                pilots.sort(key=lambda p: (-p["laps"], p["total"] or 1e9))
                for i, p in enumerate(pilots):
                    p["position"] = i + 1 if p["laps"] else None
            x = runs[0]
            out.append({"heat": n, "label": race["label"] if race else
                        " · ".join(b for b in (x.get("stage") or None,
                                               f"Round {x['round']}" if x.get("round") is not None else None,
                                               f"Race {x['race']}" if x.get("race") is not None else None) if b),
                        "start": race["start"] if race else x.get("recorded_at"), "type": race["type"] if race else "",
                        "layout": self.layout_of(x["stem"]), "pilots": pilots})
        for race in races.values():                     # flown, no analysed video
            out.append({"heat": None, "label": race["label"], "start": race["start"], "type": race["type"],
                        "layout": None, "pilots": from_race(race, [])})
        out.sort(key=lambda h: h["start"] or 0)
        return out

    def run(self, stem: str):
        r = next((x for x in self.visible_runs() if x["stem"] == stem), None)
        if not r:
            return None
        st = self._json(self._tpx(stem).runs_dir / f"{stem}.stats.json")
        race = self._json(self._tpx(stem).race_data(stem)) or {}
        r["layout"] = self.layout_of(stem)
        field = self.field(r["layout"])
        if st and field and field["n_gates"] != st["n_gates"]:
            field = rstats.field_stats([st])
        r["stats"] = st
        r["compare"] = rstats.compare(st, field) if st else None
        r["field"] = field
        r["named"] = [{"name": x["name"], "from": x["from"], "to": x["to"],
                       "best": min(rstats.named_times(st["section_grid"], x["legs"]), default=None)}
                      for x in self.named(r["layout"], st["n_gates"])] if st else []
        r["passes"] = [{"t": p["t"], "gate": p.get("gate_id", -1)} for p in race.get("passes", [])]
        w = r.get("window") or [float("-inf"), float("inf")]
        marks = (st or {}).get("lap_marks")
        if marks is None:                            # stats from before lap_marks were kept
            marks = [{"lap": l["lap"], "t0": l["t0"], "t1": l["t1"]} for l in race.get("laps", [])]
        r["lap_marks"] = [m for m in marks if m["t0"] >= w[0] and m["t1"] <= w[1]]
        r["video_url"] = f"/video/{stem}.mp4{self.q}"
        return r

    def video_file(self, stem: str) -> Path:
        """A run's video: the heat video it was analysed from, read in place.
        The path saved then may be stale (the event or track folder was moved
        or renamed since): then the same file is looked for in the event's
        folder now — <event>/<heat folder>/<file>, <event>/videos/<file>, <event>/<file>."""
        with self.day.lock:
            r = self.day.state["runs"].get(stem) or {}
        if not r.get("video_path"):
            return self.tp.videos / f"{stem}.mp4"
        saved = Path(r["video_path"])
        if saved.exists():
            return saved
        event = self.tp.dir.parent                   # the data folder is <event>/lazySpotter
        for cand in (event / saved.parent.name / saved.name, event / "videos" / saved.name, event / saved.name):
            if cand.exists():
                return cand
        return saved

    def heat(self, n: int):
        runs = [r for r in self.visible_runs() if r["heat"] == n]
        with self.day.lock:
            heat = dict(self.day.state["heats"].get(str(n)) or {})
        out = []
        for r in sorted(runs, key=lambda r: r["quad"]):
            race = self._json(self._tpx(r["stem"]).race_data(r["stem"])) or {}
            st = self._json(self._tpx(r["stem"]).runs_dir / f"{r['stem']}.stats.json")
            w = r.get("window") or [float("-inf"), float("inf")]
            off = max(r["live_start_s"], w[0]) if r.get("window") else r["live_start_s"]
            r["passes"] = [{"t": p["t"] - off, "gate": p["gate_id"]} for p in race.get("passes", [])
                           if p.get("gate_id", -1) >= 1 and w[0] <= p["t"] <= w[1]]
            r["stats"] = st
            r["video_url"] = f"/video/{r['stem']}.mp4{self.q}"
            out.append(r)
        mine = [h for h in self.heats(self.all_stats()) if h["heat"] == n]
        return {"heat": n, "info": heat.get("info"), "runs": out, "result": mine[0] if mine else None,
                "pb_laps": self.pb_laps()}

    def pilot(self, name: str):
        every = self.all_stats()
        mine = [s for s in every if s["pilot"] == name]
        if not mine:
            return None
        lays = [s["layout"] for s in mine]
        layout = max(set(lays), key=lays.count)        # a pilot flies one class's layout
        mine = [s for s in self._on(every, layout) if s["pilot"] == name]
        if not mine:
            return None
        field = self.field(layout, every)
        n = field["n_gates"]
        best_sections = []
        for i in range(n):
            vals = [s["section_best"][i] for s in mine if s["section_best"][i] is not None]
            best_sections.append(min(vals) if vals else None)
        vs = [round(b - m, 3) if b is not None and m else None
              for b, m in zip(best_sections, field.get("section_median") or [None] * n)]
        return {"pilot": name, "layout": layout, "runs": sorted(mine, key=lambda s: (s.get("heat") or 0, s["stem"])),
                "sections": field.get("sections"),
                "best_sections": best_sections, "vs_field_median": vs, "field": field}

    def schedule(self):
        """The meeting as FPVTrackside runs it (an FPVTrackside event only). The
        race FPVTrackside is recording now is known when the event is open in
        the app, and is then never read from disk."""
        event = self.tp.dir.parent                   # the data folder is <event>/lazySpotter
        if not (event / "Event.json").exists():
            return {"has": False}
        if hasattr(self.day, "race_day"):              # open in the app: it knows the race being run
            return self.day.race_day() or {"has": False}
        if self._schedule is None:
            from race_day.schedule import Schedule
            self._schedule = Schedule(event)
        ev = getattr(self.day, "event", None)
        running = getattr(ev, "running", None) if getattr(self.day, "mode", None) == "live" else None
        return self._schedule.get(running)

    def track(self):
        out = {"gates": [], "legs_s": None, "field": self.field()}
        from race_day.trackside import relocate
        mem = self._json(self.tp.gate_memory)
        root = self.tp.dir.resolve()
        if mem:
            amp = "&" + self.q[1:] if self.q else ""
            for g in mem.get("memory", []):
                # saved by full path: found again in this folder if the event was moved
                found = [relocate(p, self.tp.dir) for p in g.get("embed_imgs", []) if p]
                imgs = [f.resolve() for f in found if f is not None and str(f.resolve()).startswith(str(root))][:4]
                out["gates"].append({"gate": g["gate_id"], "type": g.get("gate_type"),
                                     "images": [f"/img?p={urllib.parse.quote(str(f.relative_to(root)))}{amp}" for f in imgs]})
        tim = self._json(self.tp.learned_timing)
        if tim:
            out["legs_s"] = list(tim.get("leg_seconds", {}).values())
        return out


class FileDay:
    """An event that isn't open in the app: its saved state (lazySpotter/
    race_day.json, re-read when it changes) in the shape DataView reads."""

    def __init__(self, data_dir: Path, name: str, kind: str):
        self.tp, self.name, self.kind = TrackPaths(Path(data_dir)), name, kind
        self.lock = threading.RLock()
        self._state, self._sig = {"runs": {}, "heats": {}}, None

    @property
    def state(self) -> dict:
        p = self.tp.race_day_state
        try:
            sig = p.stat().st_mtime_ns
        except OSError:
            return {"runs": {}, "heats": {}}
        if sig != self._sig:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                self._state = {"runs": d.get("runs") or {}, "heats": d.get("heats") or {},
                               "n_gates": d.get("n_gates")}
            except (OSError, ValueError):
                pass
            self._sig = sig
        return self._state

    def dashboard_status(self) -> dict:
        from race_day import labeling
        runs = self.state["runs"]
        learned = labeling.is_learned(self.tp)
        return {"track": self.name, "mode": self.kind, "open": False, "phase": "result" if learned else "learn",
                "learn_progress": [len(runs), 0], "n_gates": self.state.get("n_gates"),
                "queue": {"analysing": 0, "waiting": 0}, "live": {}, "pilot_info": {},
                "runs_total": len(runs), "error": None}


class Library:
    """Every event of the data folder, by id (its folder relative to the data
    folder). root(): the data folder now; active(): the app's open EventDay."""

    def __init__(self, root: Callable[[], Path], active: Callable[[], object]):
        self.root, self.active = root, active
        self._views: Dict[tuple, DataView] = {}
        self._events: List[dict] = []
        self._events_at = (None, 0.0)

    def events(self) -> List[dict]:
        import time as _t
        from race_day import picker
        root = Path(self.root())
        if self._events_at[0] != root or _t.time() - self._events_at[1] > 5:
            evs = [dict(e, kind="track") for e in picker.track_folders(root)] + \
                  [dict(e, kind="FPVTrackside event") for e in picker.fpvtrackside_events(root)]
            for e in evs:
                e["id"] = e["path"].relative_to(root).as_posix()
                e["data"] = e["path"] / "lazySpotter"
            self._events, self._events_at = evs, (root, _t.time())
        return self._events

    def _event(self, eid: str) -> Optional[dict]:
        return next((e for e in self.events() if e["id"] == eid), None)

    def view(self, eid: str) -> Optional[DataView]:
        e = self._event(eid)
        if not e:
            return None
        day = self.active()
        if day is not None and Path(day.tp.dir).resolve() == e["data"].resolve():
            key = (eid, id(day))
        else:
            day, key = None, (eid, "file")
        if key not in self._views:
            self._views = {k: v for k, v in self._views.items() if k[0] != eid}     # an event opened / closed
            self._views[key] = DataView(day or FileDay(e["data"], e["name"], e["kind"]), eid)
        return self._views[key]

    def listing(self) -> List[dict]:
        out = []
        for e in sorted(self.events(), key=lambda e: -e["time"]):
            v = self.view(e["id"])
            st = self.status(v)
            out.append({"id": e["id"], "name": e["name"], "kind": e["kind"], "time": e["time"],
                        "runs": sum(1 for r in v.visible_runs() if r.get("status") == "labelled"),
                        "learned": st["phase"] == "result", "open": bool(st.get("open"))})
        return out

    @staticmethod
    def status(v: DataView) -> dict:
        if isinstance(v.day, FileDay):
            return v.day.dashboard_status()
        st = v.day.dashboard_status() if hasattr(v.day, "dashboard_status") else v.day.status()
        return dict(st, open=True)


def make_handler(lib: Library):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data).encode(), "application/json")

        def _file(self, path: Path):
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            size = path.stat().st_size
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if (a and b) else size - 1
                end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            p = urllib.parse.unquote(u.path)
            qs = urllib.parse.parse_qs(u.query)
            try:
                if p == "/api/events":
                    return self._json({"root": str(lib.root()), "events": lib.listing()})
                view = lib.view(qs.get("e", [""])[0]) if (p.startswith("/api/") or p.startswith("/video/")
                                                           or p == "/img") else None
                if view is None and (p.startswith("/api/") or p.startswith("/video/") or p == "/img"):
                    return self._json({"error": "no such event"}, 404)
                if p in ("/", "/index.html"):
                    return self._file(STATIC / "index.html")
                if p.startswith("/static/"):
                    f = (STATIC / p[len("/static/"):]).resolve()
                    return self._file(f) if str(f).startswith(str(STATIC.resolve())) else self._send(403, b"", "text/plain")
                if p == "/api/status":
                    return self._json(lib.status(view))
                if p == "/api/runs":
                    return self._json({"runs": view.runs(), "field": view.field()})
                if p == "/api/results":
                    return self._json(view.results())
                if p == "/api/ghost":
                    r = view.ghost(qs.get("p", []))
                    return self._json(r) if r else self._json({"error": "no lap to show"}, 404)
                if p.startswith("/api/run/"):
                    r = view.run(p[len("/api/run/"):])
                    return self._json(r) if r else self._json({"error": "no such run"}, 404)
                if p.startswith("/api/heat/"):
                    return self._json(view.heat(int(p[len("/api/heat/"):])))
                if p.startswith("/api/pilot/"):
                    r = view.pilot(p[len("/api/pilot/"):])
                    return self._json(r) if r else self._json({"error": "no such pilot"}, 404)
                if p == "/api/track":
                    return self._json(view.track())
                if p == "/api/schedule":
                    return self._json(view.schedule())
                if p == "/api/race/heats":
                    return self._json({"heats": raceview.heats(view)})
                if p.startswith("/api/race/heat/"):
                    r = raceview.heat(view, int(p[len("/api/race/heat/"):]))
                    return self._json(r) if r else self._json({"error": "no such heat"}, 404)
                if p.startswith("/video/"):
                    name = Path(p).name
                    if not view.visible(Path(name).stem):          # void runs aren't served
                        return self._send(404, b"not found", "text/plain")
                    return self._file(view.video_file(Path(name).stem))
                if p == "/img":
                    q = qs.get("p", [""])[0]
                    f = (view.tp.dir / q).resolve()
                    if not str(f).startswith(str(view.tp.dir.resolve())):
                        return self._send(403, b"", "text/plain")
                    return self._file(f)
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._json({"error": str(e)}, 500)
                except Exception:
                    pass

        def do_POST(self):            # read-only dashboard
            self._send(405, b"read-only", "text/plain")

        do_PUT = do_DELETE = do_PATCH = do_POST

    return H


class Dashboard:
    """root(): the data folder now; active(): the app's open EventDay (or None)."""

    def __init__(self, root: Callable[[], Path], active: Callable[[], object], port: int = 8765):
        self.lib = Library(root, active)
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(self.lib))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}/"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
