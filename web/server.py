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

from dataset_paths import TrackPaths, data_root, list_tracks, under_root  # noqa: E402
from version import full_version  # noqa: E402

STATIC = Path(__file__).parent / "static"
DET_MODEL = ROOT / "cyn_current-20260715_best.pt"     # the detector the pass scorer was trained with
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
FOLDERS = {"learn": "learn_videos", "race": "test_videos"}
ROLE_OF = {"learn_videos": "learn", "test_videos": "race", "memory_videos": "memory", "videos": "race"}

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
    for folder in (("test_videos", "videos") if LEAGUE else ("learn_videos", "test_videos", "memory_videos", "videos")):
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
    return {"n_gates": d.get("n_gates"), "method": d.get("method", "template"),
            "learned_from": d.get("learned_from", []),
            "lap_seconds": lm.get("lap_seconds") or round(sum(d.get("leg_seconds", {}).values()), 2) or None,
            "extras": sum(p["kind"] == "extra" for p in pts)}


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
    if folder not in FOLDERS.values() or not src.is_file():
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
    if tp.gate_memory.exists():
        shutil.copy2(tp.gate_memory, tp.gate_memory.with_suffix(".json.bak"))
    shutil.copy2(tp.learned_memory, tp.gate_memory)
    relabelled = _relabel_all(tp)
    return {"n_gates": n, "lap_seconds": round(lap_s, 2), "videos": used, "relabelled": relabelled}


@app.post("/api/tracks/{track}/reset")
def reset_learning(track: str):
    """Forget the learned track: moves the learned files to learn_backups/<time>/ (the YOLO analyses stay)."""
    _admin()
    tp = _track(track)
    files = [f for f in (tp.gate_memory, tp.gate_memory.with_suffix(".json.bak"), tp.learned_memory, tp.learned_timing)
             if f.exists()]
    if not files:
        return {"moved": [], "backup": None}
    dest = tp.dir / "learn_backups" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), dest / f.name)
    return {"moved": [f.name for f in files], "backup": str(dest)}


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
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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


@app.post("/api/jobs")
async def start_job(req: Request):
    body = await req.json()
    kind, tp = body.get("kind"), _track(str(body.get("track", "")))
    py = [sys.executable, "-u"]
    if kind == "analyse":
        v = next((v for v in _videos(tp) if v["name"] == body.get("video")), None)
        if v is None:
            raise HTTPException(404, "no such video")
        if LEAGUE and v["analysed"]:
            raise HTTPException(403, "This video is already analysed.")
        if any(j["video"] == v["name"] and j["track"] == tp.dir.name and j["status"] in ("queued", "running")
               for j in JOBS.values()):
            raise HTTPException(409, "This video is already being analysed.")
        cmd = py + [str(ROOT / "extract_race.py"), "--video", str(tp.dir / v["folder"] / v["name"]),
                    "--det-model", str(DET_MODEL), "--clip-device", "mps" if _has_mps() else "cpu"]
        title, after = f"Analyse {v['name']}", None
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
    else:
        raise HTTPException(400, "kind must be analyse or learn")
    job = {"id": uuid.uuid4().hex[:8], "kind": kind, "track": tp.dir.name, "video": body.get("video"),
           "title": title, "status": "queued", "progress": 0.0, "detail": "", "log": [], "created": time.time()}
    JOBS[job["id"]] = job
    threading.Thread(target=_run_job, args=(job, cmd, after), daemon=True).start()
    return job


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
LIVE_VIEW_W = 960                       # frames wider than this are scaled down for the page
LIVE: dict = {"status": "idle", "source": None, "kind": None, "error": None, "imgsz": None, "fps": None,
              "started": None, "passes": [], "stats": None, "end": None,
              "track": None, "pilot": None, "lapmode": None, "gates": None, "timing": None, "laps": [], "lap_start": None,
              "where": "mac", "detail": None, "box": None, "recording": None}
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
    return {**LIVE, "passes": LIVE["passes"][-500:]}


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
        LIVE["lap_start"] = e
    _publish(_event_subs, e)


def _live_run(spec, imgsz: int, speed: float, stop: threading.Event, tp=None, lapmode: str = "track",
              record: Optional[Path] = None, listen: bool = False):
    global _live_spotter
    rec = None
    try:
        import online_spotter as osp
        if _live_spotter is None or _live_spotter.imgsz != imgsz:
            _live_set(status="loading")
            _live_spotter = osp.OnlineSpotter(DET_MODEL, imgsz)
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
        if listen:
            _live_set(detail=f"Waiting for the stream: send it to {LIVE['source'].removeprefix('stream sent to ')}")
        src = osp.FrameSource(spec, realtime=True, speed=speed, on_raw=rec.write if rec else None, listen=listen,
                              stop=stop)
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

        _live_spotter.run(src.start(), on_event, on_frame, stop, decoder)
        if rec is not None:
            LIVE["recording"].update(rec.close(), done=True)
            rec = None
        _live_set(status="stopped" if stop.is_set() else "ended")
    except Exception as e:                               # noqa: BLE001
        _live_set(status="error", error=str(e))
    finally:
        if rec is not None:                              # an error: still finish what was recorded
            LIVE["recording"].update(rec.close(), done=True)
            _live_set()


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
    if kind == "file":
        if tp is None:
            raise HTTPException(400, "Pick a track first.")
        v = next((v for v in _videos(tp) if v["name"] == body.get("video")), None)
        if v is None:
            raise HTTPException(404, "Pick a video of the track first.")
        spec, desc = str(tp.dir / v["folder"] / v["name"]), f"{v['name']} (file, real time)"
    elif kind == "device":
        spec = int(body.get("device", 0))
        desc = f"capture device {spec}"
    elif kind == "listen":                              # the sender (the media box) streams to this Mac
        proto = str(body.get("proto", "rtmp"))
        if proto not in LIVE_LISTEN:
            raise HTTPException(400, "proto must be rtmp or rtsp")
        spec = LIVE_LISTEN[proto].format(host="0.0.0.0")
        ips = _lan_ips()
        desc = f"stream sent to {LIVE_LISTEN[proto].format(host=ips[0] if ips else 'this-mac')}"
    else:
        raise HTTPException(400, "kind must be file, device or listen")
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
                    where=where, detail=None, box=None,
                    recording={"path": None, "track": record.parent.name, "done": False} if record else None)
        if where == "box":
            LIVE["status"] = "deploying"
    _live_set()
    if where == "box":
        threading.Thread(target=_box_run, args=(tp, Path(spec), imgsz, stride, speed, realtime, lapmode, _live_stop),
                         daemon=True).start()
    else:
        threading.Thread(target=_live_run, args=(spec, imgsz, speed, _live_stop, tp, lapmode, record, kind == "listen"),
                         daemon=True).start()
    return _live_snapshot()


# ── the media box as the compute machine (box_remote.py): this page stays the control and the display ──

_box_lock = threading.Lock()


def _box_run(tp, video: Path, imgsz: int, stride: int, speed: float, realtime: bool, lapmode: str,
             stop: threading.Event):
    import base64
    import box_remote as br
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


@app.get("/api/live/devices")
def live_devices():
    """Video capture devices, as ffmpeg lists them on a Mac (the index OpenCV opens is usually the same)."""
    _admin()
    import platform
    ff = shutil.which("ffmpeg")
    if platform.system() != "Darwin" or not ff:
        return {"devices": []}
    r = subprocess.run([ff, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                       capture_output=True, text=True, timeout=10)
    out, video = [], False
    for line in r.stderr.splitlines():
        if "video devices" in line:
            video = True
        elif "audio devices" in line:
            video = False
        elif video and (m := re.search(r"\[(\d+)\] (.+)$", line)) and not m.group(2).startswith("Capture screen"):
            out.append({"index": int(m.group(1)), "name": m.group(2).strip()})
    return {"devices": out}


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
                    if LIVE["status"] not in LIVE_ACTIVE:
                        return
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpg), jpg)
        finally:
            _frame_subs.discard(sub)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", headers=_STREAM_HEADERS)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
