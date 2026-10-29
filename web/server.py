"""
Spotter web app — the pipeline behind a local web server, the UI in the browser.

    python spotter_web.py              # http://127.0.0.1:8770
    python spotter_web.py --lan        # also reachable from other devices on the network

Modes (switched in the browser): Race, Learn (auto), Learn (manual), Live (the real-time spotter on one feed).
Long work (YOLO analysis, learning) runs as background jobs with progress.

API (all JSON unless noted):
  GET  /api/tracks                          tracks with videos and learned state
  POST /api/tracks                          {name} → create a track folder
  POST /api/tracks/{t}/upload?role=…        one chunk of a video upload (role: learn | race, name, upload_id, index, last)
  GET  /api/tracks/{t}/gates                learned gates (images, offsets) and extras
  GET  /api/tracks/{t}/runs/{stem}          passes, laps and stats of an analysed video
  GET  /api/tracks/{t}/runs/{stem}/frames   per-frame tracks + detections (compact, for overlays)
  GET  /api/tracks/{t}/marks/{stem}         saved manual marks;  POST saves them
  POST /api/tracks/{t}/manual               {stems} → learn the track from the saved marks
  POST /api/tracks/{t}/reset                forget the learned track (files moved to learn_backups/)
  GET  /api/tracks/{t}/layout               where the gates roughly are (learn_layout.py);  POST moves / re-estimates
  POST /api/tracks/{t}/remove               {folder, name} → move a video to removed_videos/ (its analysis stays)
  GET  /api/video/{t}/{folder}/{name}       the video file (range requests)
  GET  /api/img?p=…                         an image inside the data folder
  POST /api/jobs                            {kind: analyse | learn, track, video?, gates?}
  GET  /api/jobs                            all jobs (status, progress, last log lines)
  POST /api/live/start                      {kind: file | device | url, …} → the real-time spotter on one feed
  POST /api/live/stop                       stop it;  GET /api/live: its state and passes so far
  GET  /api/live/events                     server-sent events: state, every pass, stats twice a second
  GET  /api/live/mjpeg                      the feed with the tracked gates (MJPEG)
  GET  /api/live/devices                    capture devices (Mac)

League mode (spotter_web.py --league, for sharing with pilots): only race videos are listed, a video can only be
uploaded (in chunks, so it fits through a Cloudflare tunnel) and analysed; learning, marks, removing, creating
tracks and cancelling jobs are refused. --tracks limits which tracks are served.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset_paths import TrackPaths, data_root, list_tracks, track_of, under_root  # noqa: E402
from version import full_version  # noqa: E402

STATIC = Path(__file__).parent / "static"
import model_profiles  # noqa: E402
DET_MODEL = model_profiles.detector_path()   # the model picked (v2 / v3): its detector; set again when switched
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
FOLDERS = {"learn": "learn_videos", "race": "test_videos"}
ROLE_OF = {"learn_videos": "learn", "test_videos": "race", "memory_videos": "memory", "videos": "race",
           "combo_videos": "combo"}       # combo: a quad view of several pilots, split into one video per pilot
COMBO_MIN_W = 1600                      # a video this wide may be a combo view: offered ⊞ Split

LEAGUE = os.environ.get("SPOTTER_LEAGUE") == "1"
TRACKS = {t.strip() for t in os.environ.get("SPOTTER_TRACKS", "").split(",") if t.strip()} or None
UPLOAD_CHUNK_MAX = 64 << 20           # bytes per request (Cloudflare's free tunnel refuses bodies over 100 MB)
UPLOAD_MAX = 8 << 30
# league page: per video, per pilot per hour, and all pilots' uploads together (then uploads pause)
LEAGUE_VIDEO_MAX = 1 << 30
LEAGUE_PER_HOUR = 2
LEAGUE_TOTAL_MAX = 10 << 30
_uploads_lock = threading.Lock()
_started: Dict[str, dict] = {}        # upload_id → {who, t, bytes}: uploads in progress

APP_VERSION = full_version()
app = FastAPI(title="FPV spotter", version=APP_VERSION)
app.add_middleware(GZipMiddleware, minimum_size=2000)


@app.middleware("http")
async def no_stale_pages(req: Request, call_next):
    """The page, scripts and styles are re-checked on every load (cheap: unchanged files answer 304), so an
    update reaches every browser at once; videos and images may be cached."""
    resp = await call_next(req)
    if not req.url.path.startswith(("/api/video/", "/api/img")):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


# ──────────────────────────────────────────────────────────────
# Tracks and videos
# ──────────────────────────────────────────────────────────────

def _tracks() -> List[TrackPaths]:
    return [t for t in list_tracks() if TRACKS is None or t.dir.name in TRACKS]


def _admin():
    """Refuse an action pilots may not do on the shared (league) page."""
    if LEAGUE:
        raise HTTPException(403, "Not available on the league page.")


def _track(name: str) -> TrackPaths:
    tp = next((t for t in _tracks() if t.dir.name == name), None)
    if tp is None:
        raise HTTPException(404, f"no track {name!r}")
    return tp


def _video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    out = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           "fps": round(fps, 2), "duration": round(n / fps, 2) if fps else 0}
    cap.release()
    return out


_meta_cache: Dict[str, dict] = {}


def _videos(tp: TrackPaths) -> List[dict]:
    """The track's videos; on the league page only the analysed race videos and the ones uploaded since start."""
    out = []
    for folder in (("test_videos", "videos") if LEAGUE else ("learn_videos", "test_videos", "memory_videos", "videos",
                                                             "combo_videos")):
        d = tp.dir / folder
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            if LEAGUE and not tp.race_data(f.stem).exists() and \
                    not any(u["track"] == tp.dir.name and u["name"] == f.name for u in _ledger()):
                continue
            key = f"{f}:{f.stat().st_mtime}"
            if key not in _meta_cache:
                _meta_cache[key] = _video_meta(f)
            stem = f.stem
            out.append({
                "name": f.name, "stem": stem, "folder": folder, "role": ROLE_OF.get(folder, "race"),
                "bytes": f.stat().st_size,
                "url": f"/api/video/{tp.dir.name}/{folder}/{f.name}",
                "analysed": tp.race_data(stem).exists(),
                "learn_analysed": tp.learn_passes(stem).exists() or tp.race_data(stem).exists(),
                "has_marks": (tp.runs_dir / f"{stem}.marks.json").exists(),
                **_meta_cache[key],
            })
    return out


def _learned(tp: TrackPaths) -> Optional[dict]:
    if not tp.learned_timing.exists() or not tp.gate_memory.exists():
        return None
    d = json.loads(tp.learned_timing.read_text(encoding="utf-8"))
    lm = d.get("lap_model") or {}
    pts = lm.get("points", [])
    return {"n_gates": d.get("n_gates"), "method": d.get("method", "template"), "own": tp.learned_own,
            "model": model_profiles.active(),
            "learned_from": d.get("learned_from", []),
            "lap_seconds": lm.get("lap_seconds") or round(sum(d.get("leg_seconds", {}).values()), 2) or None,
            "extras": sum(p["kind"] == "extra" for p in pts), "layout": _layout_summary(tp)}


def _layout_summary(tp: TrackPaths) -> Optional[dict]:
    """The Learn window's map stage: when the track's layout was estimated, from how many clean laps."""
    import learn_layout
    try:
        d = learn_layout.load_file(tp)
    except (OSError, ValueError):
        return None
    return d and {"made": d.get("made"), "laps_used": d.get("laps_used"), "closure": d.get("closure"),
                  "moved": len(d.get("manual", {})), "method": d.get("method")}


@app.get("/api/model")
def get_model():
    """The models to pick from (model_profiles.py) and the one picked."""
    return {"active": model_profiles.active(),
            "models": [{"name": k, "label": p["label"], "about": p["about"], "detector": p["detector"],
                        "ready": model_profiles.detector_path(k).exists() and model_profiles.scorer_path(k).exists()}
                       for k, p in model_profiles.PROFILES.items()]}


@app.post("/api/model")
async def set_model(req: Request):
    """{model}: analyses, learning and Live use this model from now on (each model keeps its own analyses)."""
    global DET_MODEL
    _admin()
    name = str((await req.json()).get("model", ""))
    if name not in model_profiles.PROFILES:
        raise HTTPException(400, f"model must be one of {', '.join(model_profiles.PROFILES)}")
    if LIVE["status"] in LIVE_ACTIVE:
        raise HTTPException(409, "Live is running — stop it before switching the model.")
    if not (model_profiles.detector_path(name).exists() and model_profiles.scorer_path(name).exists()):
        raise HTTPException(400, f"Model {name}'s files aren't here ({model_profiles.PROFILES[name]['detector']}).")
    model_profiles.set_active(name)
    DET_MODEL = model_profiles.detector_path(name)
    _meta_cache.clear()
    return get_model()


@app.get("/api/tracks")
def tracks():
    return {"data_root": "" if LEAGUE else str(data_root()), "league": LEAGUE, "version": APP_VERSION,
            "upload_limits": _upload_limits() if LEAGUE else None,
            "tracks": [{"name": tp.dir.name, "videos": _videos(tp), "learned": _learned(tp)}
                       for tp in _tracks()]}


@app.post("/api/tracks")
async def create_track(req: Request):
    _admin()
    name = str((await req.json()).get("name", "")).strip()
    if not re.fullmatch(r"[\w\- ]{1,64}", name) or name == "training":
        raise HTTPException(400, "Use letters, digits, spaces, - or _ for the track name.")
    tp = TrackPaths(data_root() / name).ensure()
    for f in FOLDERS.values():
        (tp.dir / f).mkdir(exist_ok=True)
    return {"name": name}


def _free_name(folder: Path, name: str) -> Path:
    """name in folder, or name-2, name-3… so an upload never replaces a video already there."""
    dest, k = folder / name, 2
    while dest.exists():
        dest = folder / f"{Path(name).stem}-{k}{Path(name).suffix}"
        k += 1
    return dest


def _ledger_file() -> Path:
    return data_root() / "league_uploads.json"


def _ledger() -> List[dict]:
    """Pilots' uploads from the league page: [{track, name, bytes, who, t}]."""
    try:
        return json.loads(_ledger_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _league_bytes() -> int:
    """Disk taken by pilots' uploads still in the tracks (a video you removed no longer counts)."""
    tracks = {t.dir.name: t for t in list_tracks()}
    return sum(u["bytes"] for u in _ledger()
               if u["track"] in tracks and (tracks[u["track"]].dir / FOLDERS["race"] / u["name"]).exists())


def _upload_limits() -> dict:
    return {"video_gb": LEAGUE_VIDEO_MAX / 2 ** 30, "per_hour": LEAGUE_PER_HOUR,
            "paused": _league_bytes() >= LEAGUE_TOTAL_MAX}


def _who(req: Request) -> str:
    """The pilot's address (behind a Cloudflare tunnel every request comes from this computer)."""
    return req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "?")


def _league_admit(req: Request, upload_id: str, size: int, tp: TrackPaths):
    """At an upload's first chunk: refuse it if the video, the pilot's hour or the league folder is over its limit."""
    now = time.time()
    for k in [k for k, u in _started.items() if now - u["t"] > 3600]:
        del _started[k]
    for p in (tp.dir / ".uploads").glob("*.part"):            # abandoned uploads
        if now - p.stat().st_mtime > 3600:
            p.unlink(missing_ok=True)
    if size > LEAGUE_VIDEO_MAX:
        raise HTTPException(413, f"Videos up to {LEAGUE_VIDEO_MAX / 2 ** 30:.0f} GB, please.")
    who = _who(req)
    recent = sum(u.get("who") == who and now - u["t"] < 3600 for u in _ledger()) + \
        sum(u["who"] == who for k, u in _started.items() if k != upload_id)
    if recent >= LEAGUE_PER_HOUR:
        raise HTTPException(429, f"{LEAGUE_PER_HOUR} uploads per hour, please — try again later.")
    if _league_bytes() + sum(u["bytes"] for k, u in _started.items() if k != upload_id) + size > LEAGUE_TOTAL_MAX:
        raise HTTPException(507, "Uploads are paused: the league folder is full. Ask the organiser.")
    _started[upload_id] = {"who": who, "t": now, "bytes": size}


def _free_name(folder: Path, name: str) -> Path:
    """name in folder, or name-2, name-3… so an upload never replaces a video already there."""
    dest, k = folder / name, 2
    while dest.exists():
        dest = folder / f"{Path(name).stem}-{k}{Path(name).suffix}"
        k += 1
    return dest


@app.post("/api/tracks/{track}/upload")
async def upload(track: str, role: str, name: str, upload_id: str, index: int, last: int, req: Request,
                 size: int = 0):
    """One chunk of a video upload (the raw bytes as the body, chunks in order from index 0; size = the whole
    file); the last one moves the file into the track under a free name."""
    tp = _track(track)
    if role not in FOLDERS:
        raise HTTPException(400, "role must be learn or race")
    if role != "race":
        _admin()
    name = Path(name).name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "Not a video file (.mp4, .mov, .avi or .mkv).")
    if not re.fullmatch(r"[0-9a-f]{8,32}", upload_id):
        raise HTTPException(400, "bad upload id")
    part = tp.dir / ".uploads" / f"{upload_id}.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    if LEAGUE:
        with _uploads_lock:
            if index == 0:
                _league_admit(req, upload_id, size, tp)
            elif upload_id not in _started:
                raise HTTPException(400, "Upload expired — start again.")
    have = part.stat().st_size if part.exists() else 0
    if index == 0 and have:
        part.unlink()
        have = 0
    elif index > 0 and not part.exists():
        raise HTTPException(400, "upload chunk out of order")
    body = await req.body()
    limit = min(UPLOAD_MAX, max(size, 1)) if LEAGUE else UPLOAD_MAX   # no more than the size it was admitted with
    if len(body) > UPLOAD_CHUNK_MAX or have + len(body) > limit:
        part.unlink(missing_ok=True)
        _started.pop(upload_id, None)
        raise HTTPException(413, "Video too large.")
    with open(part, "ab") as f:
        f.write(body)
    if not last:
        return {"received": have + len(body)}
    dest = _free_name(tp.dir / FOLDERS[role], name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(part), dest)
    if LEAGUE:
        with _uploads_lock:
            u = _started.pop(upload_id, {})
            _ledger_file().write_text(json.dumps(_ledger() + [
                {"track": tp.dir.name, "name": dest.name, "bytes": dest.stat().st_size, "who": u.get("who", _who(req)),
                 "t": time.time()}], indent=1), encoding="utf-8")
    return {"name": dest.name, "folder": FOLDERS[role]}


@app.post("/api/tracks/{track}/remove")
async def remove_video(track: str, req: Request):
    _admin()
    """Take a video out of the track: moves it to removed_videos/<folder>/ (its analysis in runs/ stays)."""
    tp = _track(track)
    body = await req.json()
    folder, name = str(body.get("folder", "")), Path(str(body.get("name", ""))).name
    src = tp.dir / folder / name
    if folder not in (*FOLDERS.values(), "combo_videos") or not src.is_file():
        raise HTTPException(404, f"no video {folder}/{name}")
    dest = tp.dir / "removed_videos" / folder / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(f"{dest.stem}.{time.strftime('%Y%m%d-%H%M%S')}{dest.suffix}")
    shutil.move(str(src), dest)
    return {"moved_to": str(dest)}


@app.get("/api/video/{track}/{folder}/{name}")
def video(track: str, folder: str, name: str):
    tp = _track(track)
    f = tp.dir / folder / name
    if folder not in ROLE_OF or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, media_type="video/mp4")


@app.get("/api/img")
def img(p: str):
    f = Path(p)
    if not f.is_absolute():
        f = data_root() / f
    r = under_root(f)
    if r is None or not f.is_file() or (TRACKS is not None and r.parts[0] not in TRACKS):
        raise HTTPException(404)
    return FileResponse(f)


def _img_url(p: str) -> str:
    return f"/api/img?p={p}" if p else ""


# ──────────────────────────────────────────────────────────────
# Learned track, results
# ──────────────────────────────────────────────────────────────

@app.get("/api/tracks/{track}/gates")
def gates(track: str):
    tp = _track(track)
    if not tp.gate_memory.exists():
        return {"gates": [], "extras": [], "learned": None}
    mem = json.loads(tp.gate_memory.read_text(encoding="utf-8")).get("memory", [])
    out = [{"id": g["gate_id"], "type": g.get("gate_type", "?"), "offset_s": g.get("offset_s"),
            "passes": g.get("learned_passes"), "images": [_img_url(p) for p in g.get("embed_imgs", [])[:8]]}
           for g in sorted(mem, key=lambda g: g.get("order_idx", 0))]
    extras = []
    if tp.learned_timing.exists():
        lm = json.loads(tp.learned_timing.read_text(encoding="utf-8")).get("lap_model") or {}
        extras = [{"id": p["id"], "offset_s": p["offset_s"], "seen_laps": p["seen_laps"]}
                  for p in lm.get("points", []) if p["kind"] == "extra"]
        for g, p in zip(out, [p for p in lm.get("points", []) if p["kind"] == "gate"]):
            g["seen_laps"] = p["seen_laps"]
    return {"gates": out, "extras": extras, "learned": _learned(tp)}


def _race_file(tp: TrackPaths, stem: str) -> Path:
    for f in (tp.race_data(stem), tp.learn_passes(stem)):
        if f.exists():
            return f
    raise HTTPException(404, f"{stem} has not been analysed yet")


_race_cache: Dict[str, tuple] = {}


def _load_race(tp: TrackPaths, stem: str) -> dict:
    f = _race_file(tp, stem)
    key = str(f)
    mt = f.stat().st_mtime
    if key not in _race_cache or _race_cache[key][0] != mt:
        _race_cache[key] = (mt, json.loads(f.read_text(encoding="utf-8")))
    return _race_cache[key][1]


@app.get("/api/tracks/{track}/runs/{stem}")
def run(track: str, stem: str):
    from race_day.stats import run_stats
    tp = _track(track)
    race = _load_race(tp, stem)
    learned = _learned(tp)
    n = int(learned["n_gates"]) if learned and learned.get("n_gates") else \
        max((p.get("gate_id", 0) for p in race.get("passes", [])), default=0)
    passes = [{"t": p["t"], "gate_id": p.get("gate_id", -1), "source": p.get("source", ""),
               "extra_id": p.get("extra_id"), "type": p.get("gate_type", ""), "p": p.get("p_pass"),
               "reason": p.get("reason", ""), "recovered": bool(p.get("recovered")),
               "img": _img_url(p.get("query_img", ""))}
              for p in sorted(race.get("passes", []), key=lambda p: p["t"])]
    stats = run_stats(race, {}, n) if n >= 2 and race.get("laps") else None
    return {"stem": stem, "n_gates": n, "passes": passes, "laps": race.get("laps", []), "stats": stats,
            "legs": _track_legs(tp), "layout": _layout(tp),
            "is_learn_run": not tp.race_data(stem).exists(),
            "info": {k: race.get(k) for k in ("gate_id_logic", "timing", "pass_logic", "lap_seconds",
                                              "det_model", "det_device", "duration", "fps")}}


@app.get("/api/tracks/{track}/runs/{stem}/frames")
def frames(track: str, stem: str):
    """Compact per-frame data: [t, tracks, dets] with
    track = [tid, x1, y1, x2, y2, type, score, stage, area_ratio, cdist], det = [x1, y1, x2, y2, type, conf]."""
    tp = _track(track)
    race = _load_race(tp, stem)
    types, stages = {}, {}

    def ti(x, d):
        return d.setdefault(str(x), len(d))
    out = []
    for e in race.get("frames", []):
        trs = [[tr["track_id"], *tr["bbox"], ti(tr.get("type"), types), tr.get("score", 0),
                ti(tr.get("stage", "idle"), stages), tr.get("area_ratio", 0), tr.get("cdist", 0)]
               for tr in e.get("tracks", [])]
        dets = [[*d[:4], ti(d[4], types), d[5]] for d in e.get("dets", [])]
        if trs or dets:
            out.append([e["t"], trs, dets])
    return {"types": list(types), "stages": list(stages), "has_dets": any(f[2] for f in out), "frames": out}


# ──────────────────────────────────────────────────────────────
# Manual learning
# ──────────────────────────────────────────────────────────────

@app.get("/api/tracks/{track}/marks/{stem}")
def get_marks(track: str, stem: str):
    f = _track(track).runs_dir / f"{stem}.marks.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"marks": []}


@app.post("/api/tracks/{track}/marks/{stem}")
async def save_marks(track: str, stem: str, req: Request):
    _admin()
    tp = _track(track)
    marks = (await req.json()).get("marks", [])
    tp.runs_dir.mkdir(parents=True, exist_ok=True)
    (tp.runs_dir / f"{stem}.marks.json").write_text(json.dumps({"marks": marks}, indent=1), encoding="utf-8")
    return {"saved": len(marks)}


@app.post("/api/tracks/{track}/manual")
async def learn_manual(track: str, req: Request):
    """Learn the track from the marks saved on one or more analysed videos."""
    _admin()
    import gate_decoder as gd
    import lap_learner as ll
    import learn_track
    tp = _track(track)
    stems = (await req.json()).get("stems", [])
    seqs, marks, raw, used = [], [], {}, []
    for stem in stems:
        mf = tp.runs_dir / f"{stem}.marks.json"
        if not mf.exists():
            continue
        race = _load_race(tp, stem)
        ps = sorted(race.get("passes", []), key=lambda p: p["t"])
        raw[stem] = ps
        used.append(stem)
        seqs.append(ll.from_passes(ps))
        marks.append([(float(m["t"]), int(m["gate"])) for m in
                      json.loads(mf.read_text(encoding="utf-8"))["marks"]])
    if not used:
        raise HTTPException(400, "No marked video: mark the gates and save first.")
    try:
        g1, gts, extras, lap_s, n = ll.points_from_marks(seqs, marks)
    except ValueError as e:
        raise HTTPException(400, str(e))
    learn_track.save_lap_model(tp, used, raw, g1, gts, extras, lap_s, n, method="manual")
    own = tp.gate_memory_own                             # the picked model's own learned track
    if own.exists():
        shutil.copy2(own, own.with_suffix(".json.bak"))
    shutil.copy2(tp.learned_memory_own, own)
    relabelled = _relabel_all(tp)
    _layout_after_learning(tp)
    return {"n_gates": n, "lap_seconds": round(lap_s, 2), "videos": used, "relabelled": relabelled}


@app.post("/api/tracks/{track}/reset")
async def reset_learning(track: str, req: Request):
    """{analyses}: delete the picked model's learned track — its own files only (another model's, v2's shared
    one included, stay) — and with analyses: true also all its analyses (<track>/runs[_v3]/). Moved to
    learn_backups/<time>-<model>/, not deleted for good."""
    _admin()
    tp = _track(track)
    body = await req.json() if (await req.body()) else {}
    if any(j["track"] == tp.dir.name and j["status"] in ("queued", "running") for j in JOBS.values()):
        raise HTTPException(409, "A job of this track is running — wait for it to finish.")
    model = model_profiles.active()
    own = [tp.gate_memory_own, tp.gate_memory_own.with_suffix(".json.bak"), tp.learned_memory_own, tp.learned_timing_own,
           tp.gate_layout_own]
    files = [f for f in own if f.exists()]
    runs = tp.runs_dir if body.get("analyses") and tp.runs_dir.is_dir() and any(tp.runs_dir.iterdir()) else None
    if not files and runs is None:
        return {"moved": [], "backup": None, "model": model}
    dest = tp.dir / "learn_backups" / f"{time.strftime('%Y%m%d-%H%M%S')}-{model}"
    k = 2
    while dest.exists():                                 # two in one second
        dest = dest.with_name(f"{dest.name.split('~')[0]}~{k}")
        k += 1
    dest.mkdir(parents=True)
    for f in files:
        shutil.move(str(f), dest / f.name)
    if runs is not None:
        shutil.move(str(runs), dest / runs.name)
    return {"moved": [f.name for f in files] + ([runs.name + "/"] if runs else []), "backup": str(dest), "model": model}


# ──────────────────────────────────────────────────────────────
# The track's layout: where the gates roughly are (learn_layout.py), moved by hand here
# ──────────────────────────────────────────────────────────────

def _layout(tp: TrackPaths, make: bool = True) -> Optional[dict]:
    """The layout for the race map and the Layout window ({n_gates, gates: [{id, x, y, manual}], …}); estimated
    the first time it's asked for on a learned track."""
    import learn_layout
    if not tp.learned_timing.exists():
        return None
    d = learn_layout.load_file(tp)
    try:
        n = int(json.loads(tp.learned_timing.read_text(encoding="utf-8"))["n_gates"])
        if make and (d is None or d.get("n_gates") != n):
            d = learn_layout.save_estimate(tp)
    except (OSError, ValueError, KeyError) as e:          # no layout: the plain loop
        print(f"layout of {tp.dir.name}: {e}")
        return None
    if d is None:
        return None
    mem = json.loads(tp.gate_memory.read_text(encoding="utf-8")).get("memory", []) if tp.gate_memory.exists() else []
    kind = {g["gate_id"]: g.get("gate_type", "?") for g in mem}           # the 3D view models each gate by it
    return {**{k: d.get(k) for k in ("n_gates", "laps_used", "closure", "method", "made")},
            "gates": [{**g, "type": kind.get(g["id"], "?")} for g in learn_layout.gates(d)],
            "own": tp.gate_layout_own.exists()}


def _layout_after_learning(tp: TrackPaths) -> str:
    import learn_layout
    try:
        d = learn_layout.save_estimate(tp)
        return f"Layout: {d['method']} ({len(d['manual'])} gates kept where moved by hand)"
    except Exception as e:                               # noqa: BLE001
        return f"Layout not estimated: {e}"


@app.get("/api/tracks/{track}/layout")
async def get_layout(track: str):
    tp = _track(track)
    d = await asyncio.to_thread(_layout, tp)
    if d is None:
        raise HTTPException(404, "Learn the track first.")
    mem = json.loads(tp.gate_memory.read_text(encoding="utf-8")).get("memory", []) if tp.gate_memory.exists() else []
    look = {g["gate_id"]: g for g in mem}
    for g in d["gates"]:
        m = look.get(g["id"], {})
        g["img"] = _img_url(m["embed_imgs"][0]) if m.get("embed_imgs") else None
    d["legs"] = _track_legs(tp)
    return d


@app.post("/api/tracks/{track}/layout")
async def set_layout(track: str, req: Request):
    """{move: {gate id: [x, y]}} gates moved by hand · {reset: true} back to the estimate · {estimate: true} estimate
    again (the moved gates stay) · {transform: "mirror" | "rotate"} the whole layout."""
    import learn_layout
    _admin()
    tp = _track(track)
    body = await req.json()
    if not tp.learned_timing.exists():
        raise HTTPException(400, "Learn the track first.")
    if body.get("estimate") or learn_layout.load_file(tp) is None:
        await asyncio.to_thread(learn_layout.save_estimate, tp)
    d = learn_layout.load_file(tp)
    if body.get("transform") in ("mirror", "rotate"):
        d = learn_layout.transform(tp, body["transform"])
    if body.get("reset"):
        d["manual"] = {}
    for k, p in (body.get("move") or {}).items():
        if 1 <= int(k) <= d["n_gates"]:
            d.setdefault("manual", {})[str(int(k))] = [round(float(p[0]), 3), round(float(p[1]), 3)]
    tp.gate_layout_own.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return await get_layout(track)


def _relabel_all(tp: TrackPaths) -> List[str]:
    """Re-label every analysed race video with the newly learned track (no YOLO)."""
    import relabel_run
    done = []
    for f in sorted(tp.runs_dir.glob("*.race_data.json")):
        stem = f.name[: -len(".race_data.json")]
        vids = [v for v in _videos(tp) if v["stem"] == stem]
        if vids:
            try:
                relabel_run.relabel(str(tp.dir / vids[0]["folder"] / vids[0]["name"]))
                done.append(stem)
            except SystemExit:
                pass
    return done


# ──────────────────────────────────────────────────────────────
# Background jobs (analysis, auto learning)
# ──────────────────────────────────────────────────────────────

JOBS: Dict[str, dict] = {}
_job_lock = threading.Lock()          # one heavy job at a time (YOLO uses the whole GPU)


def _run_job(job: dict, cmd: List[str], after=None):
    with _job_lock:
        job["status"], job["started"] = "running", time.time()
        env = {**os.environ, "FPV_MODEL": job.get("model") or model_profiles.active()}   # the model it was started with
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env)
        job["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            if not line or "blocked by cooldown" in line:
                continue
            m = re.match(r"Progress:\s*([\d.]+)%", line)
            if m:
                job["progress"] = float(m.group(1))
                job["detail"] = line
            else:
                job["log"].append(line)
                job["log"] = job["log"][-200:]
        proc.wait()
        ok = proc.returncode == 0 and job["status"] != "cancelled"
        if ok and after:
            try:
                after(job)
            except Exception as e:                       # noqa: BLE001
                job["log"].append(f"after-step failed: {e}")
        if job["status"] != "cancelled":
            job["status"] = "done" if ok else "failed"
        job["progress"] = 100.0 if ok else job.get("progress", 0)
        job["ended"] = time.time()


def _queue_job(kind: str, tp: TrackPaths, video: Optional[str], title: str, cmd: List[str], after=None) -> dict:
    model = model_profiles.active()
    job = {"id": uuid.uuid4().hex[:8], "kind": kind, "track": tp.dir.name, "video": video, "model": model,
           "title": f"{title} · {model}" if kind in ("analyse", "learn") else title,
           "status": "queued", "progress": 0.0, "detail": "", "log": [], "created": time.time()}
    JOBS[job["id"]] = job
    threading.Thread(target=_run_job, args=(job, cmd, after), daemon=True).start()
    return job


def _analyse_job(tp: TrackPaths, v: dict) -> dict:
    cmd = [sys.executable, "-u", str(ROOT / "extract_race.py"), "--video", str(tp.dir / v["folder"] / v["name"]),
           "--det-model", str(DET_MODEL), "--clip-device", "mps" if _has_mps() else "cpu"]

    def after(job):                                      # one more analysed video: maybe more clean laps for the map
        if tp.gate_layout_own.exists() and tp.learned_timing.exists():
            job["log"].append(_layout_after_learning(tp))
    return _queue_job("analyse", tp, v["name"], f"Analyse {v['name']}", cmd, after)


@app.post("/api/jobs")
async def start_job(req: Request):
    body = await req.json()
    kind, tp = body.get("kind"), _track(str(body.get("track", "")))
    py = [sys.executable, "-u"]
    if kind == "analyse":
        v = next((v for v in _videos(tp) if v["name"] == body.get("video")), None)
        if v is None:
            raise HTTPException(404, "no such video")
        if v["folder"] == "combo_videos":
            raise HTTPException(400, "A combo view: analyse its pilots' videos (…_CH1 …).")
        if LEAGUE and v["analysed"]:
            raise HTTPException(403, "This video is already analysed.")
        if any(j["video"] == v["name"] and j["track"] == tp.dir.name and j["status"] in ("queued", "running")
               for j in JOBS.values()):
            raise HTTPException(409, "This video is already being analysed.")
        return _analyse_job(tp, v)
    elif kind == "learn":
        _admin()
        n = int(body.get("gates", 0))
        if n < 2:
            raise HTTPException(400, "number of gates?")
        if not any(v["folder"] == "learn_videos" for v in _videos(tp)):
            raise HTTPException(400, "Add at least one learn video first.")
        cmd = py + [str(ROOT / "learn_track.py"), tp.dir.name, "--gates", str(n), "--install",
                    "--det-model", str(DET_MODEL)]
        title = f"Learn {tp.dir.name} ({n} gates)"

        def after(job):
            job["log"].append("Re-labelled: " + (", ".join(_relabel_all(tp)) or "no race videos yet"))
            job["log"].append(_layout_after_learning(tp))
    elif kind == "split":
        # a combo view → one video per pilot, next to it; analyse: then each pilot's race is analysed (a video that
        # turns out not to be a combo is analysed as it is) — what an uploaded race video goes through
        analyse = bool(body.get("analyse"))
        v = next((v for v in _videos(tp) if v["name"] == body.get("video")), None)
        if v is None:
            raise HTTPException(404, "no such video")
        if LEAGUE:                                      # pilots: only a race video of theirs, not analysed yet
            if not analyse or v["analysed"] or v["folder"] not in ("test_videos", "videos"):
                raise HTTPException(403, "Not allowed on the league page.")
        else:
            _admin()
        parked = v["folder"] == "combo_videos"          # split before: its pilots' videos go to test_videos/ again
        if parked and any(x["stem"].startswith(v["stem"] + "_CH") for x in _videos(tp)):
            raise HTTPException(400, "Already split: its pilots are …_CH1 … in the video menu.")
        if any(j["video"] == v["name"] and j["track"] == tp.dir.name and j["status"] in ("queued", "running")
               for j in JOBS.values()):
            raise HTTPException(409, "This video is already being split or analysed.")
        src = tp.dir / v["folder"] / v["name"]
        out_dir = tp.dir / ("test_videos" if parked else v["folder"])
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = py + [str(ROOT / "combo_split.py"), str(src), "--out", str(out_dir)] + (["--if-combo"] if analyse else [])
        title = f"Split {v['name']}" + (" and analyse its pilots" if analyse else "")

        def after(job):
            made = [out_dir / f"{src.stem}_CH{n}.mp4" for n in range(1, 17)]
            made = [f for f in made if f.exists() and f.stat().st_mtime >= job["started"]]
            if parked:
                if analyse:
                    for f in made:
                        _analyse_job(tp, {"name": f.name, "folder": out_dir.name})
                job["log"].append("Pilots: " + ", ".join(f.name for f in made))
                return
            if not made:                                # not a combo (analyse only): the video as it is
                job["log"].append(f"{src.name} isn't a combo view: analysing it as one pilot's video.")
                _analyse_job(tp, v)
                return
            dest = _free_name(tp.dir / "combo_videos", src.name)   # out of learning / analysis
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dest)
            job["log"].append(f"The combo moved to combo_videos/{dest.name} (Live can still play one pilot of it).")
            if analyse:
                for f in made:
                    _analyse_job(tp, {"name": f.name, "folder": v["folder"]})
                job["log"].append("Analysing " + ", ".join(f.name for f in made))
    else:
        raise HTTPException(400, "kind must be analyse, learn or split")
    return _queue_job(kind, tp, body.get("video"), title, cmd, after)


@app.get("/api/tracks/{track}/combo")
async def combo_layout(track: str, video: str):
    """Is this video a combo (quad) view? Its tiles (combo_split.py; found once, then cached)."""
    tp = _track(track)
    v = next((v for v in _videos(tp) if v["name"] == video), None)
    if v is None:
        raise HTTPException(404, "no such video")
    import combo_split
    layout = await asyncio.to_thread(_combo_layout_file, tp.dir / v["folder"] / v["name"])
    return {**layout, "combo": combo_split.is_combo(layout)}


@app.get("/api/tracks/{track}/combo_race")
async def combo_race(track: str, video: str):
    """A combo view's race, for Race mode: each pilot's video split from it (<stem>_CH<k>, same clock as the combo),
    its gate passes and laps as analysed by the model picked, and the learned track for the race map."""
    tp = _track(track)
    vids = _videos(tp)
    v = next((v for v in vids if v["name"] == video), None)
    if v is None:
        raise HTTPException(404, "no such video")
    layout = await asyncio.to_thread(_combo_layout_file, tp.dir / v["folder"] / v["name"])
    learned = _learned(tp)
    pilots = []
    for tile in layout.get("tiles", []):
        pv = next((x for x in vids if x["stem"] == f"{v['stem']}_CH{tile['ch']}"), None)
        if pv is None:
            continue
        p = {"ch": tile["ch"], "tile": tile, "video": pv["name"], "analysed": pv["learn_analysed"]}
        if p["analysed"]:
            race = _load_race(tp, pv["stem"])
            race_passes = sorted((x for x in race.get("passes", []) if x.get("source") == "RACE"), key=lambda x: x["t"])
            p["passes"] = [{"t": x["t"], "gate_id": x.get("gate_id", -1), "recovered": bool(x.get("recovered"))}
                           for x in race_passes if x.get("gate_id", -1) >= 1]
            p["laps"] = [{k: L.get(k) for k in ("lap", "t0", "t1", "dt")} for L in race.get("laps", [])]
            p["end"] = race_passes[-1]["t"] if race_passes else None
        pilots.append(p)
    n = int(learned["n_gates"]) if learned and learned.get("n_gates") else \
        max((x["gate_id"] for p in pilots for x in p.get("passes", [])), default=0)
    return {"video": v["name"], "size": layout.get("size"), "grid": layout.get("grid"), "n_gates": n,
            "legs": _track_legs(tp), "layout": _layout(tp), "pilots": pilots}


@app.get("/api/jobs")
def jobs():
    out = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
    if LEAGUE:                                   # no paths of this computer on the shared page
        hide = [(str(data_root()), "<data>"), (str(ROOT), "<app>")]
        out = [{**j, "log": [_hide(l, hide) for l in j["log"]]} for j in out]
    return {"jobs": out}


def _hide(line: str, pairs) -> str:
    for a, b in pairs:
        line = line.replace(a, b)
    return line


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    _admin()
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404)
    if job["status"] in ("running", "queued") and job.get("pid"):
        import os
        import signal
        job["status"] = "cancelled"
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    return job


def _has_mps() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:                                    # noqa: BLE001
        return False


# ──────────────────────────────────────────────────────────────
# Live: the real-time spotter (online_spotter.py) on one feed, in this process. The page gets the feed with
# the tracked gates as MJPEG and every pass as an event (server-sent events) the moment it's detected.
# ──────────────────────────────────────────────────────────────

LIVE_IMGSZ = (320, 416, 512, 640)
LIVE_LISTEN = {"rtmp": "rtmp://{host}:1935/live/feed", "rtsp": "rtsp://{host}:8554/live/feed"}   # sent to us
COMBO_ALL = -1                          # Live combo: every pilot at once
MULTI_FPS = 30                          # … each analysed at this rate with the Core ML detector (4 tiles: ~14 ms) …
MULTI_FPS_SLOW = 15                     # … or this without it (PyTorch on the GPU, 4 tiles: ~70 ms)
LIVE_VIEW_W_MULTI = 1280
VIEW_FPS = 15                           # the combo's picture to the page (all pilots)
LIVE_VIEW_W = 960                       # frames wider than this are scaled down for the page
LIVE: dict = {"status": "idle", "source": None, "kind": None, "error": None, "imgsz": None, "fps": None,
              "started": None, "passes": [], "stats": None, "end": None,
              "track": None, "pilot": None, "lapmode": None, "gates": None, "timing": None, "laps": [], "lap_start": None,
              "where": "mac", "detail": None, "box": None, "recording": None, "combo": None,
              "pilots": None, "legs": None, "layout": None, "starts": []}
LIVE_ACTIVE = ("deploying", "loading", "running")
_live_lock = threading.Lock()
_live_stop: Optional[threading.Event] = None
_live_spotter = None                    # online_spotter.OnlineSpotter, kept loaded between runs
_event_subs: set = set()                # (event loop, asyncio.Queue) per page listening
_frame_subs: set = set()


def _q_put(q: asyncio.Queue, item, latest_only: bool):
    if q.full():
        if not latest_only:
            return
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(item)


def _publish(subs: set, item, latest_only: bool = False):
    """From the spotter's thread to every listening page (each on the server's event loop)."""
    for loop, q in list(subs):
        try:
            loop.call_soon_threadsafe(_q_put, q, item, latest_only)
        except RuntimeError:                             # that loop is closed
            subs.discard((loop, q))


def _live_snapshot() -> dict:
    return {**LIVE, "passes": LIVE["passes"][-500:], "preview": PREVIEW}


def _live_set(**kw):
    LIVE.update(kw)
    _publish(_event_subs, {"type": "state", **{k: v for k, v in LIVE.items() if k != "passes"}})


def _live_event(e: dict):
    """An event of the running spotter (here or on the box) → the live state and every listening page."""
    t = e.get("type")
    if t == "pass":
        LIVE["passes"].append(e)
    elif t == "gate":                                  # on the box the pass is a separate object: name it here
        p = next((p for p in reversed(LIVE["passes"]) if p.get("n") == e.get("n")), None)
        if p is not None:
            p["gate_id"], p["conf"] = e.get("gate_id"), e.get("conf")
    elif t == "stats":
        LIVE["stats"] = e
    elif t == "end":
        LIVE["end"] = e
    elif t == "lap":
        LIVE["laps"].append(e)
    elif t == "start":
        if "pilot" in e:                               # a pilot of a combo: every start kept (the race map)
            LIVE["starts"].append(e)
        else:
            LIVE["lap_start"] = e
    _publish(_event_subs, e)


def _combo_layout_file(path: Path) -> dict:
    """A video's combo layout (combo_split.py), kept next to its runs."""
    import combo_split
    tp = track_of(path)
    cache = (tp.runs_dir / f"{path.stem}.combo.json") if tp else None
    if cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return json.loads(cache.read_text())
    layout = combo_split.detect_layout(combo_split.sample_frames(str(path)))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(layout))
    return layout


def _combo_apply(src, ch: int, path: Optional[Path] = None, stop: Optional[threading.Event] = None) -> Optional[dict]:
    """Show / spot / save only channel ch of a combo view (0: the whole picture). A live feed's layout is found in
    its last whole frames (waits for them, ~2 s)."""
    import combo_split
    if not ch:
        src.crop = None
        return None
    if path is not None:
        layout = _combo_layout_file(path)
    else:
        until = time.monotonic() + 6
        while len(src.raw_samples) < 8 and time.monotonic() < until and not src.ended and not (stop and stop.is_set()):
            time.sleep(0.1)
        if not src.raw_samples:
            raise RuntimeError("no picture to find the combo's tiles in")
        layout = combo_split.detect_layout(list(src.raw_samples))
    if not combo_split.is_combo(layout):
        raise RuntimeError("This feed doesn't look like a combo view (no tile edges found) — Combo: off.")
    if ch > len(layout["tiles"]):
        raise RuntimeError(f"There's no CH{ch}: the combo is {layout['grid'][0]}×{layout['grid'][1]}.")
    tile = layout["tiles"][ch - 1]
    src.crop = tile
    return {"ch": ch, "grid": layout["grid"], "tile": tile, "active": [t["active"] for t in layout["tiles"]]}


_multi_spotter = None                   # the detector for every pilot of a combo: Core ML (Neural Engine) on a Mac


def _coreml_detector() -> Optional[Path]:
    """The detector exported to Core ML (320×192, half precision; once, into exports/): 5× faster than PyTorch on
    the GPU for a combo's four tiles, same laps (Storm: identical laps, 3 more false passes of 62)."""
    import platform
    out = ROOT / "exports" / f"{DET_MODEL.stem}_320x192.mlpackage"
    if out.exists():
        return out
    if platform.system() != "Darwin":
        return None
    try:
        _live_set(detail="Preparing the detector for the Neural Engine (once, ~1 min)…")
        from ultralytics import YOLO
        made = Path(YOLO(str(DET_MODEL)).export(format="coreml", imgsz=(192, 320), half=True, nms=False))
        out.parent.mkdir(exist_ok=True)
        shutil.move(str(made), out)
        return out
    except Exception as e:                               # noqa: BLE001 — PyTorch at a slower pace instead
        print(f"[live] Core ML export failed: {e}", file=sys.stderr)
        return None


def _combo_tiles(src, path: Optional[Path], stop: threading.Event) -> List[dict]:
    """The pilots' tiles of a combo view: a file's channels that ever have a picture; a live feed's all of them (a
    pilot's channel may be grey until the drone is powered)."""
    import combo_split
    src.crop = None
    if path is not None:
        layout = _combo_layout_file(path)
    else:
        _live_set(detail="Finding the combo's tiles…")
        until = time.monotonic() + 6
        while len(src.raw_samples) < 8 and time.monotonic() < until and not src.ended and not stop.is_set():
            time.sleep(0.1)
        if not src.raw_samples:
            raise RuntimeError("no picture to find the combo's tiles in")
        layout = combo_split.detect_layout(list(src.raw_samples))
    if not combo_split.is_combo(layout):
        raise RuntimeError("This feed doesn't look like a combo view (no tile edges found).")
    tiles = [t for t in layout["tiles"] if t["active"] or path is None]
    LIVE["combo"] = {"ch": COMBO_ALL, "grid": layout["grid"], "tiles": tiles}
    return tiles


def _track_legs(tp) -> Optional[List[float]]:
    """The learned time of each leg G1→G2 … GN→G1 (s): the race map spaces the gates by them."""
    f = tp.learned_timing if tp else None                  # the model's own, else v2's
    if not f or not f.exists():
        return None
    d = json.loads(f.read_text())
    n, legs = int(d.get("n_gates", 0)), d.get("leg_seconds", {})
    out = [legs.get(f"G{i}->G{i % n + 1}") for i in range(1, n + 1)]
    return [float(x) for x in out] if n and all(x for x in out) else None


def _live_run(spec, imgsz: int, speed: float, stop: threading.Event, tp=None, lapmode: str = "track",
              record: Optional[Path] = None, listen: bool = False, key: Optional[str] = None, combo: int = 0,
              names: Optional[List[str]] = None):
    """key: a live feed (device / stream sent to us), shown in the preview before the run: the run takes the open
    feed over and hands it back to the preview when it ends."""
    global _live_spotter
    rec = src = None
    try:
        with _pv_req_lock:
            src = _preview_take(key)                     # the preview's open feed (None: it opens its own)
        import online_spotter as osp
        if _live_spotter is None or _live_spotter.imgsz != imgsz or getattr(_live_spotter, "model_file", None) != DET_MODEL:
            _live_set(status="loading")
            _live_spotter = osp.OnlineSpotter(DET_MODEL, imgsz)
            _live_spotter.model_file = DET_MODEL
        decoder = None
        if lapmode == "track" and tp is not None and tp.gate_memory.exists():   # stage 2: gate names and laps
            import online_decoder
            decoder = online_decoder.for_track(tp)
        if decoder is not None:
            _live_spotter.clip                              # loaded now, not at the first pass
            _live_set(gates=decoder.n, timing=decoder.timing)
        if record is not None:                          # save the feed (a new track, to learn later)
            from feed_recorder import FeedRecorder
            rec = FeedRecorder(record, 30.0)            # its fps is the feed's, set below (it starts at frame 1)
        if src is None:
            if listen:
                _live_set(detail=f"Waiting for the stream: send it to {LIVE['source'].removeprefix('stream sent to ')}")
            src = osp.FrameSource(spec, realtime=True, speed=speed, listen=listen, stop=stop)
            step = src.pace(30)                          # a 60 fps feed (combo recordings): an even 30 fps
            if src.kind == "file" and combo > 0:        # the tile before the first frame
                LIVE["combo"] = _combo_apply(src, combo, Path(spec))
            if src.cap is not None:
                src.start()
        step = src.pace(30)                              # (again: a feed handed over by the preview)
        if src.kind != "file" and (combo > 0 or src.crop is not None):
            if combo > 0:
                _live_set(detail="Finding the combo's tiles…")
            LIVE["combo"] = _combo_apply(src, max(combo, 0), None, stop)
        tiles = None
        if combo == COMBO_ALL:                           # every pilot of the combo at once
            tiles = _combo_tiles(src, Path(spec) if src.kind == "file" else None, stop)
            global _multi_spotter
            if _multi_spotter is None or getattr(_multi_spotter, "model_file", None) != DET_MODEL:
                ml = _coreml_detector()
                _live_set(detail="Loading the detector…")
                _multi_spotter = osp.OnlineSpotter(ml, 320) if ml else _live_spotter
                _multi_spotter.model_file = DET_MODEL
                _multi_spotter._clip = _live_spotter._clip
            fast = _multi_spotter is not _live_spotter    # Core ML
            step = src.pace(MULTI_FPS if fast else MULTI_FPS_SLOW)
            LIVE["pilots"] = [{"ch": t["ch"], "name": (names[t["ch"] - 1] if names and t["ch"] <= len(names)
                                                         and names[t["ch"] - 1] else f"CH{t['ch']}")} for t in tiles]
            LIVE["legs"] = _track_legs(tp) if decoder is not None else None
            LIVE["layout"] = _layout(tp) if decoder is not None else None
        src.restart_clock()                              # the run's feed time from 0
        src.on_raw = rec.write if rec else None          # saved: the pilot's tile from here on
        if rec is not None:
            rec.fps = src.fps
            LIVE["recording"].update(path=str(record.relative_to(data_root())), **rec.info())
        if stop.is_set():
            return _live_set(status="stopped", detail=None)
        _live_set(status="running", fps=round(src.fps, 1), detail=None)

        def on_event(e):
            if rec is not None and e.get("type") == "stats":
                LIVE["recording"].update(rec.info())
                e["recording"] = LIVE["recording"]
            _live_event(e)
        def on_frame(frame, t, tracks, states, last_pass):
            if not _frame_subs:
                return
            img = osp.draw(frame, tracks, states, last_pass, t)
            if img.shape[1] > LIVE_VIEW_W:
                img = cv2.resize(img, (LIVE_VIEW_W, round(img.shape[0] * LIVE_VIEW_W / img.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                _publish(_frame_subs, buf.tobytes(), latest_only=True)

        if tiles is not None:
            import copy
            names_now = [p["name"] for p in LIVE["pilots"]]

            last_pic = [0.0]

            def on_frame_multi(frame, t, pilots):      # the page's picture: at most VIEW_FPS (a 2048×1152 combo
                now = time.monotonic()                  # drawn and encoded every frame takes ~8 ms of the 33)
                if not _frame_subs or now - last_pic[0] < 1.0 / VIEW_FPS:
                    return
                last_pic[0] = now
                img = osp.draw_multi(frame, t, pilots, names_now)
                if img.shape[1] > LIVE_VIEW_W_MULTI:
                    img = cv2.resize(img, (LIVE_VIEW_W_MULTI, round(img.shape[0] * LIVE_VIEW_W_MULTI / img.shape[1])),
                                     interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    _publish(_frame_subs, buf.tobytes(), latest_only=True)

            decoders = [copy.deepcopy(decoder) if decoder is not None else None for _ in tiles]
            if decoder is not None and _multi_spotter._clip is None:
                _multi_spotter._clip = _live_spotter.clip         # one CLIP for both
            _multi_spotter.run_multi(src, tiles, on_event, on_frame_multi, stop, decoders, keep_source=key is not None,
                                    frame_step=step)
        else:
            _live_spotter.run(src, on_event, on_frame, stop, decoder, keep_source=key is not None, frame_step=step)
        if rec is not None:
            LIVE["recording"].update(rec.close(), done=True)
            rec = None
        _live_set(status="stopped" if stop.is_set() else "ended")
    except Exception as e:                               # noqa: BLE001
        _live_set(status="error", error=str(e))
    finally:
        if src is not None:
            src.on_raw = None
            if src.kind != "file":
                src.stride, src.target_fps = 1, None     # the preview again: every frame
        if rec is not None:                              # an error: still finish what was recorded
            LIVE["recording"].update(rec.close(), done=True)
            _live_set()
        if key is not None and src is not None and not src.ended:   # the feed goes on in the preview
            _preview_start(key, None, False, LIVE["source"], src=src)
        elif src is not None:
            src.close()


# ── the preview: the live feed picked (capture device / stream sent to us) shows as soon as it's picked — nothing
# spotted or saved; ▶ Start takes the open feed over (a sender isn't dropped), ■ Stop of the run hands it back ──

PREVIEW: dict = {"status": "off", "key": None, "source": None, "detail": None, "error": None, "size": None,
                 "fps": None, "started": None, "combo": None}
PREVIEW_ON = ("opening", "on")
PREVIEW_FPS = 15                        # frames a second to the page
PREVIEW_IDLE_S = 60                     # no page watching for this long: the feed is closed (camera light off)
PREVIEW_FIRST_S = 6                     # no picture this long after opening: closed and opened again …
PREVIEW_TRIES = 3                       # … this many times
_pv: dict = {"thread": None, "stop": None, "handoff": None, "src": None}
_pv_lock = threading.Lock()
_pv_req_lock = threading.Lock()


def _preview_set(**kw):
    PREVIEW.update(kw)
    _publish(_event_subs, {"type": "preview", **PREVIEW})


def _preview_feed(src, stop: threading.Event, handoff: threading.Event, first_s: Optional[float] = None) -> bool:
    """Frames to the page until stopped / handed over. False: no picture within first_s seconds."""
    last, idle_since, size, t_open, got = 0.0, None, None, time.monotonic(), False
    while not stop.is_set() and not handoff.is_set():
        item = src.get(timeout=0.5)
        now = time.monotonic()
        if item is not None:
            got = True
        elif not got and first_s is not None and now - t_open > first_s:
            return False
        if not _frame_subs:                              # nobody looking
            idle_since = idle_since or now
            if now - idle_since > PREVIEW_IDLE_S:
                _preview_set(status="off", detail=None)
                return True
        else:
            idle_since = None
        if item is None:
            if src.ended:
                _preview_set(status="error", error="the feed ended")
                return True
            continue
        frame = item[0]
        if size != f"{frame.shape[1]}×{frame.shape[0]}" or PREVIEW["status"] != "on":   # (a combo's tile: smaller)
            size = f"{frame.shape[1]}×{frame.shape[0]}"
            _preview_set(status="on", size=size, fps=round(src.fps, 1), detail=None)
        if now - last < 1 / PREVIEW_FPS or not _frame_subs:
            continue
        last = now
        img = frame
        if img.shape[1] > LIVE_VIEW_W:
            img = cv2.resize(img, (LIVE_VIEW_W, round(img.shape[0] * LIVE_VIEW_W / img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            _publish(_frame_subs, buf.tobytes(), latest_only=True)
    return True


def _preview_thread(key, spec, listen, desc, stop, handoff, src=None):
    try:
        if src is None:
            import online_spotter as osp
            _preview_set(status="opening", key=key, source=desc, error=None, size=None, fps=None, started=time.time(),
                         combo=None,
                         detail=f"Waiting for the stream: send it to {desc.removeprefix('stream sent to ')}" if listen
                         else f"Opening {desc}…")
            for attempt in range(1, PREVIEW_TRIES + 1):
                src = osp.FrameSource(spec, realtime=True, listen=listen, stop=stop)
                if src.cap is None:                      # stopped while waiting for the sender
                    return
                src.start()
                _pv["src"] = src
                # a capture device that gives no picture (just released by another capture, a stalled card):
                # closed and opened again; a stream sent to us may take its time
                if _preview_feed(src, stop, handoff, None if listen else PREVIEW_FIRST_S):
                    return
                _pv["src"] = None
                src.close()
                src = None
                if attempt < PREVIEW_TRIES:
                    _preview_set(detail=f"No picture from {desc} yet — opening it again ({attempt + 1}/{PREVIEW_TRIES})…")
            return _preview_set(status="error", detail=None,
                                error=f"no picture from {desc} — is another app using it? Unplug and replug it, then pick it again")
        src._ext_stop = stop                             # handed back by a run: the preview's stop is its stop now
        _preview_set(status="on", key=key, source=desc, error=None, detail=None,
                     combo=LIVE.get("combo") if src.crop is not None else None)
        _pv["src"] = src
        _preview_feed(src, stop, handoff)
    except Exception as e:                               # noqa: BLE001
        _preview_set(status="error", error=str(e), detail=None)
    finally:
        _pv["src"] = None
        if src is not None and not handoff.is_set():
            src.close()                                  # released before anything opens the device again
        if not handoff.is_set() and PREVIEW["status"] in PREVIEW_ON:
            _preview_set(status="off", detail=None)


def _preview_start(key, spec, listen, desc, src=None):
    with _pv_lock:
        stop, handoff = threading.Event(), threading.Event()
        t = threading.Thread(target=_preview_thread, args=(key, spec, listen, desc, stop, handoff, src), daemon=True)
        _pv.update(thread=t, stop=stop, handoff=handoff, key=key)
        if src is not None:
            _pv["src"] = src
        t.start()


def _preview_take(key: Optional[str]):
    """Ends the preview. Its open feed if it's the one keyed (handed over, not closed); else None, once the old
    feed is closed (the device / port free again)."""
    with _pv_lock:
        t = _pv.get("thread")
        if t is None:
            return None
        src = _pv.get("src")
        if key is not None and _pv.get("key") == key and src is not None and not src.ended:
            _pv["handoff"].set()
        else:
            _pv["stop"].set()
            src = None
        _pv["thread"] = None
    t.join(timeout=10)                                   # its feed released (or handed over)
    if src is not None:
        _preview_set(status="off", detail=None)
    return src


def _live_source(body: dict):
    """A live feed of the Live tab: (spec, description, key, listen)."""
    kind = body.get("kind")
    if kind == "device":
        idx = int(body.get("device", 0))
        if body.get("device_id"):                        # the camera picked, wherever OpenCV numbers it now
            d = next((d for d in _camera_list() if d["id"] == body["device_id"]), None)
            if d is None:
                raise HTTPException(404, "That capture device isn't connected any more.")
            return d["index"], d["name"], f"device:{d['id']}", False
        return idx, f"capture device {idx}", f"device:{idx}", False
    if kind == "listen":                                 # the sender (the media box) streams to this Mac
        proto = str(body.get("proto", "rtmp"))
        if proto not in LIVE_LISTEN:
            raise HTTPException(400, "proto must be rtmp or rtsp")
        ips = _lan_ips()
        return (LIVE_LISTEN[proto].format(host="0.0.0.0"),
                f"stream sent to {LIVE_LISTEN[proto].format(host=ips[0] if ips else 'this-mac')}", f"listen:{proto}", True)
    raise HTTPException(400, "kind must be device or listen")


@app.post("/api/live/preview")
async def live_preview(req: Request):
    """{kind: device, device, device_id} | {kind: listen, proto}: show that feed (no spotting) until ▶ Start."""
    _admin()
    body = await req.json()
    spec, desc, key, listen = _live_source(body)
    if LIVE["status"] in LIVE_ACTIVE:
        raise HTTPException(409, "Live is running.")
    def switch():                                        # one request at a time: never two captures of one device
        with _pv_req_lock:
            if _pv.get("thread") is not None and _pv.get("key") == key and PREVIEW["status"] in PREVIEW_ON:
                return                                   # already showing it
            _preview_take(None)
            _preview_start(key, spec, listen, desc)
    await asyncio.to_thread(switch)
    return PREVIEW


@app.post("/api/live/preview/combo")
async def live_preview_combo(req: Request):
    """{combo: n}: the preview shows channel n of a combo view (0: the whole picture)."""
    _admin()
    ch = int((await req.json()).get("combo") or 0)
    src = _pv.get("src")
    if src is None or PREVIEW["status"] not in PREVIEW_ON:
        raise HTTPException(409, "No preview open.")
    try:
        info = await asyncio.to_thread(_combo_apply, src, ch)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    _preview_set(combo=info)
    return PREVIEW


@app.post("/api/live/preview/stop")
async def live_preview_stop():
    _admin()
    def stop():
        with _pv_req_lock:
            _preview_take(None)
    await asyncio.to_thread(stop)
    if PREVIEW["status"] != "off":
        _preview_set(status="off", detail=None)
    return PREVIEW


def _lan_ips() -> List[str]:
    """This Mac's addresses on the local networks (what a sender on the LAN streams to)."""
    import socket
    ips = []
    for iface in ("en0", "en1", "en2", "en3", "en4", "en5", "en6", "en7", "en8"):
        try:
            ip = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            ip = ""
        if ip and ip not in ips:
            ips.append(ip)
    if not ips:                                         # not macOS: the address the default route leaves from
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.0.2.1", 9))             # nothing is sent
                ips.append(s.getsockname()[0])
        except OSError:
            pass
    return ips


@app.get("/api/live/listen")
def live_listen():
    """The addresses a sender (the media box) can stream to, per protocol."""
    _admin()
    import socket
    hosts = _lan_ips()
    name = socket.gethostname()
    if name and not name.endswith(".local") and "." not in name:
        name += ".local"
    hosts += [name] if name else []
    return {p: [u.format(host=h) for h in hosts] for p, u in LIVE_LISTEN.items()}


@app.get("/api/live")
def live_state():
    _admin()
    return _live_snapshot()


@app.post("/api/live/start")
async def live_start(req: Request):
    """{kind: file, video} | {kind: device, device, pilot} | {kind: listen, proto, pilot}; track, imgsz, speed (a file's
    pace). pilot: the name called out with the lap times of a live feed. lapmode: track (gate names + laps from the
    learned track) | off (passes only).
    With a learned track, passes are named and laps timed (stage 2); without one, passes only."""
    global _live_stop
    _admin()
    body = await req.json()
    kind = body.get("kind")
    tp = _track(str(body["track"])) if body.get("track") else None
    key = None
    if kind == "file":
        if tp is None:
            raise HTTPException(400, "Pick a track first.")
        v = next((v for v in _videos(tp) if v["name"] == body.get("video")), None)
        if v is None:
            raise HTTPException(404, "Pick a video of the track first.")
        spec, desc = str(tp.dir / v["folder"] / v["name"]), f"{v['name']} (file, real time)"
    elif kind in ("device", "listen"):
        spec, desc, key, _ = _live_source(body)
    else:
        raise HTTPException(400, "kind must be file, device or listen")
    combo = COMBO_ALL if body.get("combo") in ("all", COMBO_ALL) else int(body.get("combo") or 0)
    if not (combo == COMBO_ALL or 0 <= combo <= 16):
        raise HTTPException(400, "combo: a channel 1…, all (every pilot), or 0 (the whole picture)")
    if combo:
        desc += " · all pilots" if combo == COMBO_ALL else f" · CH{combo}"
    names = [re.sub(r"\s+", " ", str(x)).strip()[:30] for x in (body.get("pilots") or [])][:16]
    imgsz = int(body.get("imgsz", 320))
    if imgsz not in LIVE_IMGSZ:
        raise HTTPException(400, f"imgsz must be one of {LIVE_IMGSZ}")
    speed = min(4.0, max(0.25, float(body.get("speed", 1.0))))
    lapmode = str(body.get("lapmode", "track"))
    if lapmode not in ("track", "off"):
        raise HTTPException(400, "lapmode must be track or off")
    where = str(body.get("where", "mac"))
    if where not in ("mac", "box"):
        raise HTTPException(400, "where must be mac or box")
    stride = int(body.get("stride", 2 if where == "box" else 1))
    if stride not in (1, 2, 3):
        raise HTTPException(400, "stride must be 1, 2 or 3")
    realtime = str(body.get("pace", "realtime")) != "box"
    if where == "box" and combo:
        raise HTTPException(400, "The media box runs a whole video (no combo channel).")
    if where == "box" and kind != "file":
        raise HTTPException(400, "The media box runs a video file of the track (no live feed to it yet).")
    if where == "box":
        desc = f"{v['name']} on the media box"
    record = None
    if body.get("record"):                              # save the live feed for a new track
        if kind == "file" or where == "box":
            raise HTTPException(400, "Saving is for a live feed (capture device or stream) on this Mac.")
        name = re.sub(r"\s+", " ", str(body.get("record_track", ""))).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,59}", name):
            raise HTTPException(400, "Name the new track (letters, digits, space, . _ -) to save the feed under it.")
        who = re.sub(r"[^A-Za-z0-9._-]+", "_", str(body.get("pilot") or "").strip())[:30] or "feed"
        from dataset_paths import recordings_dir
        record = recordings_dir() / name / f"{who}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    with _live_lock:
        if LIVE["status"] in LIVE_ACTIVE:
            raise HTTPException(409, "Live is already running — stop it first.")
        _live_stop = threading.Event()
        LIVE.update(status="loading", source=desc, kind=kind, error=None, imgsz=imgsz, fps=None, started=time.time(),
                    passes=[], stats=None, end=None, track=tp.name if tp else None, gates=None, timing=None,
                    laps=[], lap_start=None, speed=speed if kind == "file" else 1.0,
                    pilot=str(body.get("pilot") or "").strip()[:40] or None if kind != "file" else None, lapmode=lapmode,
                    where=where, detail=None, box=None, combo=None, pilots=None, legs=None, starts=[],
                    recording={"path": None, "track": record.parent.name, "done": False} if record else None)
        if where == "box":
            LIVE["status"] = "deploying"
    _live_set()
    if where == "box":
        threading.Thread(target=_box_run, args=(tp, Path(spec), imgsz, stride, speed, realtime, lapmode, _live_stop),
                         daemon=True).start()
    else:
        threading.Thread(target=_live_run, args=(spec, imgsz, speed, _live_stop, tp, lapmode, record, kind == "listen",
                                                 key, combo, names), daemon=True).start()
    return _live_snapshot()


# ── the media box as the compute machine (box_remote.py): this page stays the control and the display ──

_box_lock = threading.Lock()


def _box_run(tp, video: Path, imgsz: int, stride: int, speed: float, realtime: bool, lapmode: str,
             stop: threading.Event):
    import base64
    import box_remote as br
    _preview_take(None)                                  # the box plays a file: no preview meanwhile
    box = br.Box()
    errlog: List[str] = []
    proc = None
    try:
        with _box_lock:
            _live_set(detail=f"connecting to {box.host}…")
            st = box.status()
            if not st.get("reachable"):
                raise RuntimeError(f"can't reach the media box at {box.host}: {st.get('error', '')[-200:]}")
            _live_set(box={"host": box.host, "model": st.get("model")})
            models = br.export_models(DET_MODEL, imgsz, lambda m: _live_set(detail=m))
            named = lapmode == "track" and tp is not None and tp.gate_memory.exists()
            paths = box.deploy(tp.dir if named else None, tp.name, video, models, lambda m: _live_set(detail=m))
            if not named:
                paths.pop("track", None)
            if stop.is_set():
                return _live_set(status="stopped", detail=None)
            _live_set(status="loading", detail="starting the spotter on the box…")
            proc = box.start(paths, imgsz, stride, speed, realtime=realtime)

        def watch_stop():
            stop.wait()
            box.stop()
        threading.Thread(target=watch_stop, daemon=True).start()

        def on_event(e):
            t = e.get("type")
            if t == "frame":
                if _frame_subs:
                    _publish(_frame_subs, base64.b64decode(e["jpg"]), latest_only=True)
            elif t == "status":
                _live_set(detail=e.get("detail"))
            elif t == "ready":
                _live_set(status="running", fps=round(float(e.get("fps") or 0), 1), gates=e.get("gates"),
                          timing=e.get("timing"), detail=None,
                          box={**(LIVE.get("box") or {}), "stride": e.get("stride"), "realtime": realtime})
            else:
                _live_event(e)

        def on_log(line):
            print(f"[box] {line}", flush=True)
            errlog.append(line)
            del errlog[:-12]
        br.read_events(proc, on_event, on_log)
        if stop.is_set():
            _live_set(status="stopped", detail=None)
        elif LIVE.get("end"):
            _live_set(status="ended", detail=None)
        else:
            raise RuntimeError("the box's spotter stopped: " + " | ".join(errlog[-4:]))
    except Exception as e:                               # noqa: BLE001
        _live_set(status="error", error=str(e)[-400:], detail=None)
    finally:
        if proc is not None and proc.poll() is None:
            box.stop()


@app.get("/api/live/box")
def live_box_config():
    _admin()
    import box_remote as br
    return br.get_config()


@app.post("/api/live/box")
async def live_box_set(req: Request):
    """{host, user}: where the media box is."""
    _admin()
    import box_remote as br
    body = await req.json()
    host = str(body.get("host", "")).strip()
    if not host or not re.fullmatch(r"[A-Za-z0-9._:-]+", host):
        raise HTTPException(400, "The box's host name or IP address")
    user = str(body.get("user", "")).strip()
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise HTTPException(400, "The SSH user name on the box")
    return br.set_config(host, user)


@app.get("/api/live/box/find")
def live_box_find():
    """Media boxes on the network (the box's own discovery)."""
    _admin()
    import box_remote as br
    return {"boxes": br.discover()}


@app.get("/api/live/box/status")
def live_box_status():
    _admin()
    import box_remote as br
    try:
        return br.Box().status()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/live/box/clean")
def live_box_clean():
    """Delete everything this app put on the box (its temporary folder)."""
    _admin()
    if LIVE["where"] == "box" and LIVE["status"] in LIVE_ACTIVE:
        raise HTTPException(409, "Stop the run on the box first.")
    import box_remote as br
    with _box_lock:
        try:
            return br.Box().clean()
        except RuntimeError as e:
            raise HTTPException(400, str(e))


@app.post("/api/live/stop")
def live_stop():
    _admin()
    if _live_stop is not None:
        _live_stop.set()
    return _live_snapshot()


def _camera_list() -> List[dict]:
    """The Mac's cameras with the index OpenCV opens each by: OpenCV (AVFoundation) numbers them in the order of
    their unique IDs — not ffmpeg's order (there the FaceTime camera came first and a USB capture card second,
    the other way round in OpenCV)."""
    import platform
    if platform.system() != "Darwin":
        return []
    try:
        r = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=15)
        cams = json.loads(r.stdout or "{}").get("SPCameraDataType", [])
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    cams = sorted((c for c in cams if c.get("spcamera_unique-id")), key=lambda c: c["spcamera_unique-id"])
    return [{"index": i, "name": str(c.get("_name", f"camera {i}")), "id": c["spcamera_unique-id"]}
            for i, c in enumerate(cams)]


@app.get("/api/live/devices")
def live_devices():
    """Video capture devices (Mac), with the index OpenCV opens and a unique ID that stays with the device."""
    _admin()
    return {"devices": _camera_list()}


_STREAM_HEADERS = {"Cache-Control": "no-cache", "Content-Encoding": "identity", "X-Accel-Buffering": "no"}


@app.get("/api/live/events")
async def live_events():
    _admin()
    sub = (asyncio.get_running_loop(), asyncio.Queue(maxsize=1000))
    _event_subs.add(sub)

    async def gen():
        try:
            yield f"data: {json.dumps({'type': 'snapshot', **_live_snapshot()})}\n\n"
            while True:
                try:
                    e = await asyncio.wait_for(sub[1].get(), 15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(e)}\n\n"
        finally:
            _event_subs.discard(sub)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_STREAM_HEADERS)


@app.get("/api/live/mjpeg")
async def live_mjpeg():
    _admin()
    sub = (asyncio.get_running_loop(), asyncio.Queue(maxsize=1))
    _frame_subs.add(sub)

    async def gen():
        try:
            while True:
                try:
                    jpg = await asyncio.wait_for(sub[1].get(), 5)
                except asyncio.TimeoutError:
                    if LIVE["status"] not in LIVE_ACTIVE and PREVIEW["status"] not in PREVIEW_ON:
                        return
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpg), jpg)
        finally:
            _frame_subs.discard(sub)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", headers=_STREAM_HEADERS)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
