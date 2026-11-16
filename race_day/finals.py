"""
The finals in FPVTrackside: every class's mains in the same rounds, one race
per main in turn.

    Finals 1   A W1-1 · B W1-1 · C W1-1 · Beginners A W1-1 · A W1-2 · B W1-2 …
    Finals 2   A W2-1 · B W2-1 · … (as their line-ups become known)

Each main is a Seeder stage (race_day/eventsetup.py, "Make the brackets"),
made without the Seeder's own sending: the Seeder puts a stage's races in a
round of its own, and the club flies A, B, C, D, Beginners, then A again. So
the app does the Seeder's two jobs itself:

  1. the next Finals round (only the one after the round being flown)
     appears as soon as a pilot is known for one of its races: all of its
     races at once, in turn, each with the pilots known so far (the bracket
     diagram's seats: the 3rd and 4th of W2-1 are in L2-1 the moment W2-1 is
     flown); after each race the pilots now known join their race. All at
     once because FPVTrackside keeps races in the order they're made: a race
     added to a round later would go last. Finals n holds the races of round
     n in their stages (a chase the ace's 2nd, 3rd… race go in the rounds
     after, once the Seeder says they're needed; one it no longer needs is
     deleted, unless flown). Within a round, the mains take turns: the 1st
     race of each main (classes in their order, mains A, B, C…), then the
     2nd of each…
  2. when such a race finishes in FPVTrackside, its finishing order goes to
     its stage (POST /stages/{id}/races/{race}/result), so the Seeder moves
     the pilots on; a marshal's later change in FPVTrackside (a new finishing
     order, the race's end time unchanged) is sent again while the Seeder
     still takes it, and is a warning once it doesn't.

This runs by itself after each race (race_day/eventday.py), once every class
that has pilots has its brackets made. What it made is in event_setup.json:

    made.finals.races   {"<stage id>:<stage race id>": {ext, round, fpv, class, main,
                          group, advance, title, posted}}

Each race keeps a seat per pilot on the pilot's own channel when it's free
(seat_by_channel).
"""

import hashlib
import json
from typing import Callable, Dict, List, Optional, Tuple

from race_day.eventsetup import FIN_BASE, EventSetup, seat_by_channel
from race_day.ftapi import ApiError

DONE = ("finished", "void")


def _names(lst) -> List[str]:
    return [p if isinstance(p, str) else (p.get("name") or "") for p in (lst or [])]


class Finals:
    def __init__(self, setup: EventSetup, api, log: Callable[[str], None] = print,
                 saved: Optional[Callable[[str], Optional[float]]] = None):
        """saved(fpv race id) → when FPVTrackside last saved that race (its Race.json's modified time), or
        None when unknown: a finished race is only read again after it's saved."""
        self.setup, self.api, self.log, self.saved = setup, api, log, saved
        self._sent_hash = None
        self.status = ""
        self.warnings: List[str] = []

    # ── what's made ────────────────────────────────────────────

    def mains(self) -> List[Tuple[str, str, int]]:
        """(class, main, stage id) in running order: classes as in the setup, mains A, B, C…"""
        made = self.setup.data["made"]["stages"]
        return [(c["name"], m, made[c["name"]][m]) for c in self.setup.classes
                if c["name"] in made for m in sorted(made[c["name"]])]

    def waiting_for(self) -> List[str]:
        """Classes with pilots whose brackets aren't made yet (the finals wait for them)."""
        made = self.setup.data["made"]["stages"]
        return [c["name"] for c in self.setup.classes if c["pilots"] and not made.get(c["name"])]

    @property
    def races(self) -> Dict[str, dict]:
        return self.setup.data["made"]["finals"].setdefault("races", {})

    def by_fpv(self, fpv_id: str) -> Optional[dict]:
        """Our record of a finals race, by its FPVTrackside race ID (for the announcer)."""
        fpv_id = (fpv_id or "").lower()
        return next((r for r in self.races.values() if (r.get("fpv") or "").lower() == fpv_id), None) \
            if fpv_id else None

    # ── one pass ───────────────────────────────────────────────

    def tick(self) -> bool:
        """Results in, then the ready races out. True when something changed."""
        self.setup.reload()
        mains = self.mains()
        if not mains or self.waiting_for():
            self.status = (f"finals: waiting for the brackets of {', '.join(self.waiting_for())}"
                           if mains else "")
            return False
        stages = {sid: self.api.stage(sid) for _, _, sid in mains}
        changed = self._results(stages)
        if changed:
            stages = {sid: self.api.stage(sid) for _, _, sid in mains}
        changed |= self._place(mains, stages)
        changed |= self._sync(stages)
        if changed:
            self.setup.save()
        self._status(mains, stages)
        return changed

    def _results(self, stages: Dict[int, dict]) -> bool:
        """Finished FPVTrackside races → their stage's result. The finishing order is compared with the one
        sent, not the race's end time: a marshal's lap edit changes the order but not the end, and the
        Seeder must move the pilots by the corrected order. A race is only read again when FPVTrackside
        saved it since the last look (`saved`, else every look). A changed order the Seeder no longer
        takes (the next race already flown) is a warning: the bracket must be fixed by hand."""
        mine = {r["fpv"].lower(): (k, r) for k, r in self.races.items() if r.get("fpv")}
        if not mine:
            return False
        fpv = {(x.get("id") or "").lower(): x for x in self.api.get("/races?state=finished") or []}
        changed = False
        for fid, (key, rec) in mine.items():
            race = fpv.get(fid)
            if not race:
                continue
            sid, rid = key.split(":", 1)
            sr = next((r for r in stages.get(int(sid), {}).get("races") or [] if r.get("id") == rid), None)
            if not sr or sr.get("state") == "void":
                continue
            stamp = self.saved(fid) if self.saved else None
            if sr.get("state") == "finished" and rec.get("posted") and stamp is not None \
                    and rec.get("posted_stamp") == stamp:
                continue                                  # not saved since the order was last compared
            res = sorted(self.api.get(f"/races/{fid}/results") or [], key=lambda x: x.get("position") or 99)
            placement = [x.get("name") for x in res if x.get("name")]
            if not placement:
                continue
            if placement == rec.get("posted"):
                if rec.get("posted_stamp") != stamp or rec.get("posted_end") != race.get("end"):
                    rec["posted_stamp"], rec["posted_end"] = stamp, race.get("end")
                    changed = True
                if rec.pop("stale", None):
                    changed = True
                continue
            if sr.get("state") == "finished" and not sr.get("canEdit"):
                if rec.get("stale") != placement:         # warn once per new order
                    rec["stale"] = placement
                    self.warn(f"{rec.get('class')} {rec.get('main')} {rec.get('title') or rid}: the result changed "
                              f"in FPVTrackside to {', '.join(placement)} (was {', '.join(rec.get('posted') or [])}), "
                              f"but the next race is already flown — fix this bracket by hand in the Seeder")
                    changed = True
                continue
            again = bool(rec.get("posted"))
            try:
                self.api._call("POST", f"/api/v1/stages/{sid}/races/{rid}/result", {"placement": placement})
                self.log(f"finals: {rec.get('class')} {rec.get('main')} {rec.get('title') or rid} → "
                         + ", ".join(placement) + (f" (corrected, was {', '.join(rec['posted'])})" if again else ""))
            except ApiError as e:
                self.log(f"finals: {rec.get('class')} {rec.get('main')} {rid}: the Seeder refused the result ({e})")
                continue                                  # tried again at the next look
            rec["posted"], rec["posted_end"], rec["posted_stamp"] = placement, race.get("end"), stamp
            rec.pop("stale", None)
            changed = True
        return changed

    def warn(self, msg: str):
        """A bracket the app couldn't correct: logged and kept for the event window (warnings)."""
        self.log(f"finals: ⚠ {msg}")
        self.warnings = (self.warnings + [msg])[-10:]

    @staticmethod
    def finals_round(st: dict) -> Dict[str, int]:
        """Stage race id → its Finals round: its round in the stage, except a chase the ace's 2nd, 3rd… race,
        which go in the rounds after (one race of a main per round: the other mains fly in between)."""
        out, k = {}, 0
        for sr in st.get("races") or []:
            n = int(sr.get("round") or 1)
            if "chase" in (sr.get("group") or "").lower():
                n, k = n + k, k + 1
            out[sr.get("id")] = n
        return out

    @staticmethod
    def known(st: dict) -> Dict[str, List[str]]:
        """Stage race id → the pilots known for it so far, in seat order: the whole line-up of a ready race, else
        the pilots the bracket diagram has already sent there (the 3rd and 4th of W2-1 go to L2-1 the moment
        W2-1 is flown). A chase the ace's 2nd, 3rd… race only once the Seeder says it's needed (ready): the
        diagram shows the chase's pilots in every one of them from the start."""
        seats: Dict[str, List[str]] = {}
        for g in st.get("diagrams") or []:
            for n in g.get("nodes") or []:
                seats[n.get("id")] = [x.get("name") for x in n.get("seats") or [] if x.get("name")]
        out, k = {}, 0
        for sr in st.get("races") or []:
            rid = sr.get("id")
            chase = "chase" in (sr.get("group") or "").lower()
            if sr.get("state") == "ready" or sr.get("state") in DONE:
                out[rid] = _names(sr.get("pilots"))
            elif chase and k > 0:
                out[rid] = []
            else:
                out[rid] = seats.get(rid) or _names(sr.get("pilots"))
            k += 1 if chase else 0
        return out

    def _place(self, mains, stages: Dict[int, dict]) -> bool:
        """The next Finals round, as soon as a pilot is known for one of its races: all of its races at once, in
        turn (FPVTrackside keeps races in the order they're made, so a race added to a round later would go
        last), each with the pilots known so far; the others join after each race (_sync). A round already made
        is never added to."""
        fin = self.setup.data["made"]["finals"]
        done = max([r["round"] for r in self.races.values()] or [0])
        todo: Dict[int, List[Tuple[str, str, int, dict]]] = {}
        for cls, main, sid in mains:
            st = stages[sid]
            fr, kn = self.finals_round(st), self.known(st)
            last = max([int(r.get("round") or 0) for r in st.get("races") or []
                        if not any(w in (r.get("group") or "").lower() for w in ("chase", "grand", "final"))] or [0])
            wins = (st.get("setup") or {}).get("aceWinsRequired")
            for sr in st.get("races") or []:
                key = f"{sid}:{sr.get('id')}"
                if key in self.races or sr.get("state") in DONE:
                    continue
                n = max(fr[sr.get("id")], done + 1)        # a race of a round already made: in the next one
                todo.setdefault(n, []).append((cls, main, sid, dict(
                    sr, _wins=wins, _known=kn.get(sr.get("id")),
                    _semi=int(sr.get("round") or 0) == last and "loser" in (sr.get("group") or "").lower())))
        if not todo:
            return False
        n = min(todo)
        state = {f"{sid}:{sr.get('id')}": sr.get("state") for sid, st in stages.items() for sr in st.get("races") or []}
        flying = [r["round"] for k, r in self.races.items() if not r.get("void") and state.get(k) not in DONE]
        if flying and n > min(flying) + 1:                 # only the round after the one being flown
            return False
        if not any(sr["_known"] for *_, sr in todo[n]):
            return False
        for cls, main, sid, sr in todo[n]:
            k = int(fin.get("next") or 1)
            fin["next"] = k + 1
            self.races[f"{sid}:{sr.get('id')}"] = {
                "ext": FIN_BASE + 1000 + k, "round": n, "fpv": None, "class": cls, "main": main,
                "group": sr.get("group"), "advance": sr.get("expectedWinners"), "title": sr.get("title"),
                "ace_wins": sr.get("_wins"), "seats": {}, "rid": sr.get("id"),
                "stage_round": int(sr.get("round") or 0), "semi": sr.get("_semi")}
        self.log(f"finals: Finals {n} — {len(todo[n])} races, "
                 f"{sum(len(sr['_known']) for *_, sr in todo[n])} pilots known so far")
        return True

    def plan(self, stages: Dict[int, dict]) -> dict:
        """Every finals round with its races, the mains taking turns, each with the pilots known so far. A pilot
        keeps the channel they were first given in the race (rec["seats"]); a pilot who joins later gets a
        free one."""
        order = {(c, m): i for i, (c, m, _) in enumerate(self.mains())}
        live: Dict[str, dict] = {}
        known: Dict[str, List[str]] = {}
        for sid, st in stages.items():
            kn = self.known(st)
            for sr in st.get("races") or []:
                live[f"{sid}:{sr.get('id')}"] = sr
                known[f"{sid}:{sr.get('id')}"] = kn.get(sr.get("id")) or []
        per_round: Dict[int, Dict[Tuple[str, str], List[Tuple[str, dict]]]] = {}
        for key, rec in self.races.items():
            if rec.get("void") or key not in live:
                continue
            per_round.setdefault(rec["round"], {}).setdefault((rec["class"], rec["main"]), []).append((key, rec))
        seat = self._seater()
        rounds = []
        for n in sorted(per_round):
            lanes = [sorted(v, key=lambda kr: kr[1]["ext"]) for k, v in
                     sorted(per_round[n].items(), key=lambda kv: order.get(kv[0], 99))]
            races = []
            for i in range(max(len(l) for l in lanes)):
                for lane in lanes:                       # one race per main in turn
                    if i < len(lane):
                        key, rec = lane[i]
                        races.append({"externalId": rec["ext"], "pilots": self._seats(rec, known[key], seat)})
            rounds.append({"externalId": FIN_BASE + n, "eventType": "Race", "name": f"Finals {n}", "races": races})
        return {"rounds": rounds, "removeMissing": False}

    @staticmethod
    def _seats(rec: dict, pilots: List[str], seat) -> List[dict]:
        """The race's seats: the channels already given kept, the new pilots on free ones."""
        given = {p: c for p, c in (rec.get("seats") or {}).items() if p in pilots}
        fresh = seat([p for p in pilots if p not in given], taken=set(given.values()))
        out = [{"pilot": p, "channel": given[p]} if p in given else next(x for x in fresh if x["pilot"] == p)
               for p in pilots]
        rec["seats"] = {x["pilot"]: x["channel"] for x in out if x.get("channel")}
        return out

    def _seater(self):
        try:
            channels = (self.api.get("/event") or {}).get("channels") or []
            own = {(p.get("name") or "").lower(): (p.get("channel") or {}).get("label")
                   for p in self.api.pilots()}
        except ApiError:
            channels, own = [], {}
        seats = list(self.setup.data.get("channels") or [])
        return lambda pilots, taken=(): seat_by_channel(pilots, own, channels, seats, taken)

    def _void(self, stages: Dict[int, dict]) -> bool:
        """A race made in FPVTrackside that the bracket no longer needs (a chase the ace decided early): deleted
        there, unless it has been flown."""
        changed = False
        for key, rec in self.races.items():
            sid, rid = key.split(":", 1)
            sr = next((r for r in stages.get(int(sid), {}).get("races") or [] if r.get("id") == rid), None)
            if rec.get("void") or not sr or sr.get("state") != "void":
                continue
            rec["void"] = True
            changed = True
            if rec.get("fpv"):
                try:
                    race = self.api.get(f"/races/{rec['fpv']}") or {}
                    if race.get("state") == "pending":
                        self.api._call("DELETE", f"/api/v1/races/{rec['fpv']}")
                        self.log(f"finals: {rec['class']} {rec['main']} {rid} isn't needed — taken out of "
                                 f"Finals {rec['round']}")
                except ApiError as e:
                    self.log(f"finals: {rec['class']} {rec['main']} {rid} isn't needed, but: {e}")
        return changed

    def _sync(self, stages: Dict[int, dict]) -> bool:
        changed = self._void(stages)
        plan = self.plan(stages)
        if not plan["rounds"]:
            return changed
        h = hashlib.sha1(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        if h == self._sent_hash and all(r.get("fpv") for r in self.races.values() if not r.get("void")):
            return changed
        res = self.api.meeting_sync(plan, dry_run=False)
        self._sent_hash = h
        ids = {x.get("externalId"): x.get("id") for x in res.get("races") or [] if x.get("id")}
        for rec in self.races.values():
            fid = ids.get(rec["ext"])
            if fid and (rec.get("fpv") or "").lower() != fid.lower():
                rec["fpv"] = fid.lower()
                changed = True
        for w in res.get("warnings") or []:
            self.log(f"finals: ⚠ {w}")
        changed |= self._lineups(plan)
        return changed or bool(res.get("changed"))

    def _lineups(self, plan: dict) -> bool:
        """Each race not flown yet holds exactly the pilots planned (sync may only add): any other line-up is
        replaced (PUT /races/{id}/pilots), and a pilot who joined is logged."""
        by_ext = {r["ext"]: r for r in self.races.values()}
        fpv = {(x.get("id") or "").lower(): x for x in self.api.get("/races?state=pending") or []}
        changed = False
        for rnd in plan["rounds"]:
            for race in rnd["races"]:
                rec = by_ext.get(race["externalId"])
                cur = fpv.get((rec or {}).get("fpv") or "")
                if not cur:
                    continue
                want = [x["pilot"] for x in race["pilots"]]
                have = [x.get("name") for x in cur.get("pilots") or []]
                if sorted(want) != sorted(have):
                    self.api._call("PUT", f"/api/v1/races/{rec['fpv']}/pilots", race["pilots"])
                    changed = True
                if want != rec.get("shown"):
                    new = [p for p in want if p not in (rec.get("shown") or [])]
                    if new:
                        self.log(f"finals: {rec['class']} {rec['main']} {rec.get('title') or ''} "
                                 f"(Finals {rec['round']}): {', '.join(new)} in — {len(want)} of the race known")
                    rec["shown"] = want
                    changed = True
        return changed

    def _status(self, mains, stages):
        state = {f"{sid}:{sr.get('id')}": sr.get("state") for sid, st in stages.items() for sr in st.get("races") or []}
        live = {k: r for k, r in self.races.items() if not r.get("void")}
        made = sum(1 for r in live.values() if r.get("fpv"))
        left = sum(1 for v in state.values() if v not in DONE)
        rounds = len({r["round"] for r in live.values()})
        part = [f"{r['class']} {r['main']} in Finals {r['round']} ({len(r.get('shown') or [])} pilots so far)"
                for k, r in live.items() if r.get("fpv") and state.get(k) == "waiting"]
        self.status = (f"finals: {made} races in {rounds} round{'s' if rounds != 1 else ''} in FPVTrackside · "
                       + (f"{left} still to fly or come" if left else "all flown")
                       + (" · not full yet, don't start: " + "; ".join(part) if part else ""))
