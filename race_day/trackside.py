"""
An FPVTrackside event folder, read without disturbing FPVTrackside.

FPVTrackside keeps one folder per heat inside the event folder:

    <event>/Event.json, Pilots.json, Rounds.json, Stages.json
    <event>/<heat id>/N.mp4               a 2×2 grid (one quadrant per channel); N is the
                                          video source's number in FPVTrackside (2, 0, …)
                      N.recordinfo.xml    channel → quadrant, and the clock time of every frame
                      M.mp4 + .xml        a second grid (DJI feeds) when the heat had DJI pilots
                      Race.json           pilots, channels, timing detections and laps

race_day's own data lives in <event>/lazySpotter/ — nothing else in the event
folder is ever written, except lap data in a heat's Race.json (write_laps).

Being gentle with FPVTrackside (it may run on the same computer):
  - refresh() only lists folders and stat()s files; it never opens a file
    FPVTrackside may still be writing
  - while a race is running (a video still growing, or a recording without
    its final recordinfo.xml) only that heat's folder is looked at
  - a heat's files are opened once the heat is finished and they haven't
    changed for STABLE_S seconds; every read is open–read–close, and a
    failed read is simply retried on the next refresh
"""

import json
import os
import shutil
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SUBDIR = "lazySpotter"     # all of the app's data, in the event / track folder
OLD_SUBDIR = "race_day"    # where it was before 2026-10-22 (only the Race.json backups are carried over)
STABLE_S = 10.0          # a file must be unchanged this long before it's opened
NEVER = "0001/01/01"     # FPVTrackside's "no time" (a heat that wasn't flown)

# files looked at in every heat folder (stat only): Race.json and the videos,
# named by their video source's number (2.mp4, 0.mp4…)
_HEAT_FILE = re.compile(r"^(Race\.json|\d+\.mp4|\d+\.recordinfo\.xml(\.bak)?)$")
_EVENT_FILES = ("Event.json", "Pilots.json", "Rounds.json", "Stages.json")


def parse_time(s: Optional[str]) -> Optional[float]:
    """'2026/10/09 12:05:38.479' (local time) → epoch seconds; FPVTrackside's
    empty time (0001/01/01) → None."""
    if not s or s.startswith(NEVER):
        return None
    d, _, t = s.partition(" ")
    h, m, sec = (t.split(":") + ["0", "0"])[:3]
    whole = float(sec or 0)
    # the whole local time at once: on a day the clocks change, midnight + hours is an hour off
    local = datetime.strptime(d, "%Y/%m/%d").replace(hour=int(h), minute=int(m), second=int(whole))
    return local.timestamp() + (whole - int(whole))


def parse_span(s: Optional[str]) -> Optional[float]:
    """FPVTrackside's time span ('00:02:00', '00:00:00.5000000') → seconds."""
    try:
        h, m, sec = (s or "").split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except ValueError:
        return None


def _iso_time(s: str) -> float:
    """recordinfo's '2026-10-09T12:05:31.74954+03:00' → epoch seconds (the
    wall clock FPVTrackside shows, the same frame as parse_time)."""
    s = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], s)   # Python 3.9 wants 6 digits
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None).timestamp()


def read_json(path: Path):
    """Open–read–close; None if the file is missing, locked or half written."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        d = json.loads(data.decode("utf-8-sig"))
    except (OSError, ValueError):
        return None
    return d[0] if isinstance(d, list) and len(d) == 1 and isinstance(d[0], dict) and "ID" in d[0] \
        and path.name in ("Event.json", "Race.json") else d


def _stat(p: Path):
    try:
        st = p.stat()
        return (st.st_size, st.st_mtime)
    except OSError:
        return None


@dataclass
class Quad:
    channel: str                   # display name: R1, R6, DR8…
    video: Path                    # the heat's video file (read in place)
    rect: Tuple[float, float, float, float]   # x, y, w, h — fractions of the frame
    kind: str                      # "analog" or "dji" (a grid of DJI feeds)


@dataclass
class Heat:
    id: str                        # the heat folder's name (FPVTrackside's race ID)
    dir: Path
    start: float                   # epoch seconds
    end: float
    stage: str
    round: Optional[int]
    race: Optional[int]
    pilots: Dict[str, str]         # channel → pilot name
    quads: List[Quad]
    number: int = 0                # order in the day (1 = first heat flown)
    sig: tuple = ()                # file signature it was read with
    race_type: str = ""            # the round's EventType: Race, TimeTrial, Practice…
    target_laps: Optional[int] = None
    round_name: str = ""           # the round's own name, if it has one ("Open Qualifying 2", "Winners Round 1")

    @property
    def short(self) -> str:        # stable short id for file names
        return Path(self.id).stem if "." in self.id else self.id[:8]

    @property
    def label(self) -> str:
        bits = [self.stage or None, f"R{self.round}" if self.round is not None else None,
                f"race {self.race}" if self.race is not None else None]
        return " · ".join(b for b in bits if b)

    def stem(self, channel: str) -> str:
        return f"{self.short}_{channel}"


@dataclass
class _Folder:
    sig: tuple = ()                # (name, size, mtime) of the files that exist
    changed_at: float = 0.0        # our clock when sig last changed
    files: Dict[str, tuple] = field(default_factory=dict)


def relocate(path, data_dir) -> Optional[Path]:
    """A file the app saved by its full path (gate pictures, run crops), found
    again after the event or track folder was moved or renamed: the part
    after lazySpotter/ in the folder's data_dir now. None if it's nowhere."""
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in (SUBDIR, OLD_SUBDIR):
            q = Path(data_dir).joinpath(*parts[i + 1:])
            return q if q.exists() else None
    return None


def _carry_backups(old: Path, new: Path):
    """The original Race.json of every heat the app changed, from the old data
    folder: Restore keeps working, and a later change doesn't back up an
    already-changed Race.json as the original. Nothing else is carried over."""
    if not old.is_dir():
        return
    new.mkdir(parents=True, exist_ok=True)
    for f in old.iterdir():
        if f.is_file() and not (new / f.name).exists():
            shutil.copy2(f, new / f.name)


class Event:
    def __init__(self, folder):
        self.dir = Path(folder)
        if not (self.dir / "Event.json").exists():
            raise FileNotFoundError(f"not an FPVTrackside event folder (no Event.json): {self.dir}")
        self.data_dir = self.dir / SUBDIR
        _carry_backups(self.dir / OLD_SUBDIR / "backups", self.data_dir / "backups")
        self.name = ""
        self.heats: Dict[str, Heat] = {}
        self.running: Optional[str] = None       # heat id of a race being recorded now
        self._folders: Dict[str, _Folder] = {}
        self._event_sig: Dict[str, tuple] = {}
        self._meta_at = 0.0
        self.channels: Dict[str, str] = {}       # channel ID → display name
        self.pilot_names: Dict[str, str] = {}    # pilot ID → name
        self.rounds: Dict[str, dict] = {}
        self.stages: Dict[str, str] = {}
        # the event's race rules (Event.json): min_lap, race_length (s), holeshot (the first crossing isn't a lap)
        self.rules: Dict[str, object] = {"min_lap": None, "race_length": None, "holeshot": True}
        self._frames: Dict[Tuple[str, str], tuple] = {}
        self._timing: Dict[str, tuple] = {}             # heat id → (sig, timing)
        self._channel_db = self._load_channel_db()
        self._read_event_files(force=True)

    # ── event-level files ────────────────────────────────────────

    def _load_channel_db(self) -> Dict[str, dict]:
        """FPVTrackside's channel list (band, number, prefix), next to its events
        folder; optional — display names fall back to Event.json."""
        for p in (self.dir.parent.parent / "httpfiles" / "Channels.json",
                  self.dir.parent.parent / "data" / "Channels.json"):
            d = read_json(p)
            if isinstance(d, list):
                return {c["ID"]: c for c in d if "ID" in c}
        return {}

    def _read_event_files(self, force: bool = False):
        now = time.time()
        sigs = {n: _stat(self.dir / n) for n in _EVENT_FILES}
        if not force:
            if sigs == self._event_sig:
                return
            if any(s and now - s[1] < STABLE_S for s in sigs.values()):
                return                              # FPVTrackside is saving: next time
        ev = read_json(self.dir / "Event.json")
        pilots = read_json(self.dir / "Pilots.json")
        rounds = read_json(self.dir / "Rounds.json")
        stages = read_json(self.dir / "Stages.json")
        if not isinstance(ev, dict):
            return
        self.name = ev.get("Name") or self.dir.name
        self.rules = {"min_lap": parse_span(ev.get("MinLapTime")), "race_length": parse_span(ev.get("RaceLength")),
                      "holeshot": (ev.get("PrimaryTimingSystemLocation") or "Holeshot") == "Holeshot",
                      "pb_laps": ev.get("PBLaps") if isinstance(ev.get("PBLaps"), int) else None}
        ids, names = ev.get("Channels") or [], ev.get("ChannelDisplayNames") or []
        self.channels = {i: (names[k] if k < len(names) and names[k] else
                             self._channel_db.get(i, {}).get("DisplayName", i[:8]))
                         for k, i in enumerate(ids)}
        if isinstance(pilots, list):
            self.pilot_names = {p["ID"]: p.get("Name") or "?" for p in pilots if "ID" in p}
        if isinstance(rounds, list):
            self.rounds = {r["ID"]: r for r in rounds if "ID" in r}
        if isinstance(stages, list):
            self.stages = {s["ID"]: s.get("Name") or "" for s in stages if "ID" in s}
        self._event_sig = sigs

    def channel_name(self, channel_id: str) -> str:
        if channel_id in self.channels:
            return self.channels[channel_id]
        c = self._channel_db.get(channel_id)
        return c.get("DisplayName") if c else channel_id[:8]

    def _recordinfo_channel(self, text: str) -> Optional[str]:
        """recordinfo's channel text ('Raceband 1', 'DJIO4 R8') → display name."""
        text = text.strip()
        if not text or text == "None":
            return None
        for cid, disp in self.channels.items():
            c = self._channel_db.get(cid)
            if c:
                pre = (c.get("ChannelPrefix") or "").replace("\x00", "")
                if text == f"{c['Band']} {pre}{c['Number']}":
                    return disp
        band, _, rest = text.partition(" ")      # fallback: first letter of the band + the rest
        return f"{band[:1].upper()}{rest}"           # Raceband 1 → R1, DJIO4 R8 → DR8

    # ── heat folders: stat only ──────────────────────────────────

    def refresh(self) -> dict:
        """Look at the event folder (stat only). Returns {"running": heat id or
        None, "new": [heat ids], "changed": [heat ids]}. While a race is
        running only that heat's folder is looked at."""
        now = time.time()
        full = not self.running
        if self.running:
            names = [self.running]
        else:
            try:
                names = [e.name for e in os.scandir(self.dir) if e.is_dir() and e.name not in (SUBDIR, OLD_SUBDIR)]
            except OSError:
                return {"running": self.running, "new": [], "changed": []}
        running = None
        ready = []
        for name in names:
            d = self.dir / name
            files = {}
            try:
                for e in os.scandir(d):
                    if _HEAT_FILE.match(e.name):
                        try:
                            st = e.stat()
                            files[e.name] = (st.st_size, st.st_mtime)
                        except OSError:
                            pass
            except OSError:
                pass
            sig = tuple(sorted((k, v) for k, v in files.items()))
            f = self._folders.get(name)
            if f is None:                            # first sight: trust the files' own times
                newest = max((v[1] for v in files.values()), default=0.0)
                f = self._folders[name] = _Folder(sig, min(now, newest), files)
            elif sig != f.sig:
                f.sig, f.changed_at, f.files = sig, now, files
            recording = any(k.endswith(".mp4") for k in files) and now - f.changed_at < STABLE_S
            unfinished = any(k.endswith(".recordinfo.xml.bak") and k[:-4] not in files
                             and k.split(".")[0] + ".mp4" in files for k in files) and now - f.changed_at < 120
            if recording or unfinished:
                running = name
            elif "Race.json" in files and any(k.endswith(".mp4") and k[:-4] + ".recordinfo.xml" in files
                                              for k in files) and now - f.changed_at >= STABLE_S:
                ready.append(name)
        self.running = running
        new, changed = [], []
        if full:
            for h in [h for h in self.heats if h not in names]:   # heat folder deleted in FPVTrackside
                self.heats.pop(h, None)
                self._folders.pop(h, None)
        if running is None:
            self._read_event_files()
            for name in ready:
                h = self.heats.get(name)
                if h and h.sig == self._folders[name].sig:
                    continue
                heat = self._read_heat(name)
                if heat is None:
                    continue
                (changed if h else new).append(name)
                self.heats[name] = heat
            self._number()
        return {"running": running, "new": new, "changed": changed}

    def _number(self):
        for i, h in enumerate(sorted(self.heats.values(), key=lambda h: h.start), 1):
            h.number = i

    def _read_heat(self, name: str) -> Optional[Heat]:
        d = self.dir / name
        race = read_json(d / "Race.json")
        if not isinstance(race, dict) or not race.get("Valid", True):
            return None
        start, end = parse_time(race.get("Start")), parse_time(race.get("End"))
        if start is None or end is None or end <= start:
            return None                              # not flown (or not finished)
        rnd = self.rounds.get(race.get("Round"), {})
        pilots = {self.channel_name(pc["Channel"]): self.pilot_names.get(pc["Pilot"], "?")
                  for pc in race.get("PilotChannels") or [] if pc.get("Channel") and pc.get("Pilot")}
        quads = []
        for fn in sorted((k for k in self._folders[name].files if k.endswith(".mp4")), key=lambda k: int(k[:-4])):
            if (d / fn).exists() and (d / f"{fn[:-4]}.recordinfo.xml").exists():
                quads += self._read_bounds(d / f"{fn[:-4]}.recordinfo.xml", d / fn)
        if not quads:
            return None
        return Heat(id=name, dir=d, start=start, end=end, stage=self.stages.get(rnd.get("Stage"), ""),
                    round=rnd.get("RoundNumber"), race=race.get("RaceNumber"), pilots=pilots, quads=quads,
                    sig=self._folders[name].sig, race_type=rnd.get("EventType") or "",
                    target_laps=race.get("TargetLaps"), round_name=rnd.get("Name") or "")

    def _read_bounds(self, xml: Path, video: Path) -> List[Quad]:
        """Channel → quadrant from the head of a recordinfo.xml (before its frame list).
        A grid of DJI channels ('DJIO4 R8') is a "dji" video, any other "analog"."""
        try:
            with open(xml, "rb") as f:
                head = f.read(64 * 1024).decode("utf-8", "ignore")
        except OSError:
            return []
        head = head.split("<FrameTimes>")[0]
        found = []
        for b in re.findall(r"<VideoBounds>(.*?)</VideoBounds>", head, re.S):
            ch = re.search(r"<Channel>([^<]*)</Channel>", b)
            vals = [re.search(rf"<{k}>([^<]*)</{k}>", b) for k in ("X", "Y", "Width", "Height")]
            if not ch or not all(vals):
                continue
            name = self._recordinfo_channel(ch.group(1))
            if name:
                found.append((ch.group(1).strip(), name, tuple(float(v.group(1)) for v in vals)))
        kind = "dji" if found and all(text.upper().startswith("DJI") for text, _, _ in found) else "analog"
        return [Quad(name, video, rect, kind) for _, name, rect in found]

    # ── per-heat data (only for finished heats) ──────────────────

    def frame_times(self, video: Path) -> Tuple[np.ndarray, np.ndarray]:
        """(video seconds, clock seconds) for every frame of a heat video, from
        its recordinfo.xml. A video's own timestamps match 'Seconds' (±1 ms)."""
        xml = video.with_name(video.stem + ".recordinfo.xml")
        key = (str(xml), str(_stat(xml)))
        if key not in self._frames:
            with open(xml, "rb") as f:
                text = f.read().decode("utf-8", "ignore")
            rows = re.findall(r"<Time>([^<]*)</Time>\s*<Seconds>([^<]*)</Seconds>", text)
            secs = np.array([float(s) for _, s in rows])
            clock = np.array([_iso_time(t) for t, _ in rows])
            self._frames = {k: v for k, v in self._frames.items() if not k[0] == str(xml)}
            self._frames[key] = (secs, clock)
        return self._frames[key]

    def to_clock(self, video: Path, t):
        secs, clock = self.frame_times(video)
        return np.interp(t, secs, clock)

    def to_video(self, video: Path, clock_t):
        secs, clock = self.frame_times(video)
        return np.interp(clock_t, clock, secs)

    def race_json(self, heat: Heat) -> Optional[dict]:
        return read_json(heat.dir / "Race.json")

    def timing(self, heat: Heat, race: Optional[dict] = None) -> Dict[str, List[dict]]:
        """Per channel: the timing system's valid detections, in time order —
        [{"t": clock s, "holeshot": bool, "id", "type": LapRF/Manual, "lap": FPVTrackside's lap number}]."""
        if race is None:
            hit = self._timing.get(heat.id)
            if hit and hit[0] == heat.sig:
                return hit[1]
        race = race or self.race_json(heat) or {}
        out: Dict[str, List[dict]] = {}
        for det in race.get("Detections") or []:
            if not det.get("Valid"):
                continue
            t = parse_time(det.get("Time"))
            if t is None:
                continue
            out.setdefault(self.channel_name(det.get("Channel", "")), []).append(
                {"t": t, "holeshot": bool(det.get("IsHoleshot")), "id": det.get("ID"),
                 "type": det.get("TimingSystemType"), "lap": det.get("LapNumber")})
        for v in out.values():
            v.sort(key=lambda d: d["t"])
        self._timing[heat.id] = (heat.sig, out)
        return out

    def rejected(self, heat: Heat) -> Dict[str, List[float]]:
        """Per channel: the crossings FPVTrackside doesn't count (a lap under the
        minimum lap time, after the race, one a marshal disqualified) — clock s."""
        out: Dict[str, List[float]] = {}
        for det in (self.race_json(heat) or {}).get("Detections") or []:
            t = parse_time(det.get("Time"))
            if not det.get("Valid") and t is not None:
                out.setdefault(self.channel_name(det.get("Channel", "")), []).append(t)
        return {c: sorted(v) for c, v in out.items()}


# ──────────────────────────────────────────────────────────────
# Writing laps back (the only change race_day makes to the event)
# ──────────────────────────────────────────────────────────────

def format_time(t: float) -> str:
    """Epoch seconds → FPVTrackside's '2026/10/09 12:05:38.479' (local time;
    hour without a leading zero, milliseconds without trailing zeros)."""
    lt = time.localtime(t)
    ms = int(round((t % 1) * 1000))
    if ms == 1000:
        return format_time(float(int(t) + 1))
    frac = f".{ms:03d}".rstrip("0") if ms else ""
    return f"{lt.tm_year:04d}/{lt.tm_mon:02d}/{lt.tm_mday:02d} {lt.tm_hour}:{lt.tm_min:02d}:{lt.tm_sec:02d}{frac}"


def apply_changes(race: dict, changes: List[dict], channel_ids: Dict[str, str]) -> dict:
    """Apply lap changes to a Race.json dict the way an FPVTrackside marshal does:

      {"op": "add", "channel": "R6", "t": clock s}        → a Manual detection
           (TimingSystemType Manual, ValidityType Auto, Valid, Peak 0) + its lap
      {"op": "remove", "detection_id": "…"}               → that detection becomes
           Valid false, ValidityType ManualOverride (it stays in the file)

    then every changed pilot's valid laps are renumbered (holeshot = lap 0,
    RaceSector = lap × 100) and their lap entries' start/end/length redone.
    channel_ids: display name → channel ID. Returns the new dict."""
    import copy
    import uuid
    race = copy.deepcopy(race)
    dets, laps = race.setdefault("Detections", []), race.setdefault("Laps", [])
    pilot_of = {pc["Channel"]: pc["Pilot"] for pc in race.get("PilotChannels") or []}
    touched = set()
    for ch in changes:
        if ch["op"] == "remove":
            for d in dets:
                if d["ID"] == ch["detection_id"]:
                    d["Valid"], d["ValidityType"] = False, "ManualOverride"
                    touched.add(d["Pilot"])
        elif ch["op"] == "add":
            cid = channel_ids[ch["channel"]]
            pid = pilot_of.get(cid)
            if not pid:
                raise ValueError(f"no pilot on {ch['channel']} in this heat")
            did = str(uuid.uuid4())
            dets.append({"TimingSystemIndex": 0, "Channel": cid, "Time": format_time(ch["t"]), "Peak": 0,
                         "TimingSystemType": "Manual", "Pilot": pid, "LapNumber": 0, "Valid": True,
                         "ValidityType": "Auto", "IsLapEnd": True, "RaceSector": 0, "IsHoleshot": False,
                         "ID": did, "ExternalID": 0})
            laps.append({"Detection": did, "LengthSeconds": 0.0, "LapNumber": 0, "StartTime": "", "EndTime": "",
                         "ID": str(uuid.uuid4()), "ExternalID": 0})
            touched.add(pid)
    start = parse_time(race.get("Start"))
    lap_of = {l["Detection"]: l for l in laps}
    for pid in touched:
        valid = sorted((d for d in dets if d["Pilot"] == pid and d["Valid"]), key=lambda d: parse_time(d["Time"]))
        prev = start
        for n, d in enumerate(valid):
            t = parse_time(d["Time"])
            d["LapNumber"], d["RaceSector"], d["IsHoleshot"] = n, n * 100, n == 0
            lap = lap_of.get(d["ID"])
            if lap is not None:
                lap.update(LapNumber=n, StartTime=format_time(prev), EndTime=d["Time"],
                           LengthSeconds=round(t - prev, 7))
            prev = t
    return race


def write_race(heat_dir: Path, race: dict, backup_dir: Path) -> None:
    """Write a heat's Race.json safely: the original is backed up once (never
    overwritten), the new file is written beside it and renamed into place,
    then read back to check. Raises if the file is busy — retry later."""
    src = Path(heat_dir) / "Race.json"
    backup_dir.mkdir(parents=True, exist_ok=True)
    bak = backup_dir / f"{Path(heat_dir).name}.Race.json"
    if not bak.exists():
        with open(src, "rb") as f:
            original = f.read()
        bak.write_bytes(original)
    raw = read_json_raw(src)
    out = [race] if isinstance(raw, list) else race         # keep FPVTrackside's own layout
    tmp = src.with_name("Race.json.race_day.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, src)
    back = read_json(src)
    if back is None or len(back.get("Detections", [])) != len(race.get("Detections", [])):
        raise IOError("Race.json didn't read back as written")


def restore_race(heat_dir: Path, backup_dir: Path) -> bool:
    """Put the original Race.json back (from the first backup)."""
    bak = backup_dir / f"{Path(heat_dir).name}.Race.json"
    if not bak.exists():
        return False
    src = Path(heat_dir) / "Race.json"
    tmp = src.with_name("Race.json.race_day.tmp")
    tmp.write_bytes(bak.read_bytes())
    os.replace(tmp, src)
    return True


def read_json_raw(path: Path):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8-sig"))
    except (OSError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────
# Replays: a folder of loose videos (read in place, never copied)
# ──────────────────────────────────────────────────────────────

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
QUAD_RECTS = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 0.5, 0.5), (0.0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)]


class VideoFolder:
    """The same interface as Event, for a folder of videos without
    FPVTrackside: each video is a heat. What a video is (single pilot or 2×2,
    heat number, pilot names) is asked once and kept in race_day/videos.json:

        {"<file name>": {"kind": "single" | "2x2", "heat": 3,
                         "channels": ["R1", "R3", "R6", "R8"],     # 2×2: quadrant order TL TR BL BR
                         "pilots": {"R1": "naco", …}}}              # single: {"P1": "naco"}
    """

    has_timing = False

    def __init__(self, folder):
        self.dir = Path(folder)
        self.data_dir = self.dir / SUBDIR
        self.name = self.dir.name
        self.heats: Dict[str, Heat] = {}
        self.running = None
        self.channels: Dict[str, str] = {}
        self.unconfigured: List[str] = []         # videos waiting for the user to say what they are
        self._sig = {}

    @property
    def _info_path(self) -> Path:
        return self.data_dir / "videos.json"

    def info(self) -> Dict[str, dict]:
        return read_json(self._info_path) or {}

    def set_info(self, name: str, info: dict):
        self.data_dir.mkdir(exist_ok=True)
        d = self.info()
        d[name] = info
        tmp = self._info_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, self._info_path)

    def refresh(self) -> dict:
        now = time.time()
        try:
            files = [e for e in os.scandir(self.dir) if e.is_file() and Path(e.name).suffix.lower() in VIDEO_EXT]
        except OSError:
            return {"running": None, "new": [], "changed": []}
        info = self.info()
        new, seen = [], set()
        self.unconfigured = []
        for e in sorted(files, key=lambda e: e.name):
            st = e.stat()
            if now - st.st_mtime < STABLE_S and not self._complete(e.name, info):   # still being copied in
                continue
            seen.add(e.name)
            if e.name not in info:
                self.unconfigured.append(e.name)
                continue
            if e.name in self.heats and self._sig.get(e.name) == (st.st_size, st.st_mtime, str(info[e.name])):
                continue
            self.heats[e.name] = self._heat(e.name, Path(e.path), info[e.name], st.st_mtime)
            self._sig[e.name] = (st.st_size, st.st_mtime, str(info[e.name]))
            new.append(e.name)
        for gone in [h for h in self.heats if h not in seen]:
            self.heats.pop(gone)
        for h in self.heats.values():
            h.number = int(info.get(h.id, {}).get("heat") or 0)
        return {"running": None, "new": new, "changed": []}

    def _complete(self, name: str, info: Dict[str, dict]) -> bool:
        """A file known to be complete (no need to wait for it to settle)."""
        return False

    @staticmethod
    def _heat(name: str, path: Path, info: dict, mtime: float) -> Heat:
        pilots = dict(info.get("pilots") or {})
        if info.get("kind") == "2x2":
            chans = info.get("channels") or ["Q1", "Q2", "Q3", "Q4"]
            quads = [Quad(c, path, r, "analog") for c, r in zip(chans, QUAD_RECTS)]
        else:
            ch = next(iter(pilots), "P1")
            quads = [Quad(ch, path, (0.0, 0.0, 1.0, 1.0), "analog")]
            pilots = pilots or {"P1": "?"}
        return Heat(id=name, dir=path.parent, start=mtime, end=mtime, stage=path.name, round=None,
                    race=None, pilots=pilots, quads=quads, number=int(info.get("heat") or 0))

    # no timing system and no clock: video time is the time
    def timing(self, heat, race=None):
        return {}

    def rejected(self, heat):
        return {}

    def race_json(self, heat):
        return None

    def to_clock(self, video, t):
        return t

    def to_video(self, video, t):
        return t

    def frame_times(self, video):
        raise NotImplementedError


LIVE_SUBDIR = SUBDIR            # a track's data, next to its videos/ folder


class TrackVideos(VideoFolder):
    """A track's videos — recorded by lazy_spotter_v2's live capture, or opened
    in Replays: <track>/videos/ (a folder without one: its own videos), with
    all of the app's data in <track>/lazySpotter/ (videos.json — what each
    video is —, runs/, the learned gates and timing, race_day.json). Learning
    data is only ever taken from there: a track is learned once in the app.
    Recordings are described as they're made; other videos are asked about
    once. `running` is set while the app records, so race_day pauses."""

    def __init__(self, track_dir):
        track_dir = Path(track_dir)
        vids = track_dir / "videos"
        if not vids.is_dir() and any(p.suffix.lower() in VIDEO_EXT for p in track_dir.glob("*")):
            vids = track_dir                     # a plain folder of videos
        super().__init__(vids)
        self.track_dir = track_dir
        self.data_dir = track_dir / LIVE_SUBDIR
        self.name = track_dir.name

    def refresh(self) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        r = super().refresh()
        r["running"] = self.running
        return r

    def _complete(self, name: str, info: Dict[str, dict]) -> bool:
        return bool(info.get(name, {}).get("seconds"))     # the app's own recording, moved in when done

    @staticmethod
    def _heat(name: str, path: Path, info: dict, mtime: float) -> Heat:
        h = VideoFolder._heat(name, path, info, mtime)
        if info.get("seconds"):
            s = float(info["seconds"])
            h.stage = f"{path.stem}  {int(s // 60)}:{int(s % 60):02d}"
        if info.get("recorded_at"):
            h.start, h.end = float(info["recorded_at"]), float(info["recorded_at"]) + float(info.get("seconds") or 0)
        return h

    def next_heat(self) -> int:
        """The number for the next recording: after every heat known so far."""
        nums = [int(i.get("heat") or 0) for i in self.info().values()]
        for f in self.dir.glob("heat_*.*"):
            try:
                nums.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(nums, default=0) + 1


Event.has_timing = True
