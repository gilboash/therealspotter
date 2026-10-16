"""
Spotter web app — the pipeline behind a local web server, the UI in the browser.

    python spotter_web.py              # http://127.0.0.1:8770
    python spotter_web.py --lan        # also reachable from other devices on the network

Modes (switched in the browser): Race, Learn (auto), Learn (manual).
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

League mode (spotter_web.py --league, for sharing with pilots): only race videos are listed, a video can only be
uploaded (in chunks, so it fits through a Cloudflare tunnel) and analysed; learning, marks, removing, creating
tracks and cancelling jobs are refused. --tracks limits which tracks are served.
"""

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
from fastapi.responses import FileResponse
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


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
