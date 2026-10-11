"""
Spotter web app — the pipeline behind a local web server, the UI in the browser.

    python spotter_web.py              # http://127.0.0.1:8770
    python spotter_web.py --lan        # also reachable from other devices on the network

Modes (switched in the browser): Race, Learn (auto), Learn (manual).
Long work (YOLO analysis, learning) runs as background jobs with progress.

API (all JSON unless noted):
  GET  /api/tracks                          tracks with videos and learned state
  POST /api/tracks                          {name} → create a track folder
  POST /api/tracks/{t}/upload?role=…        multipart video upload (role: learn | race)
  POST /api/tracks/{t}/import               {path, role} → copy a video from this computer
  GET  /api/tracks/{t}/gates                learned gates (images, offsets) and extras
  GET  /api/tracks/{t}/runs/{stem}          passes, laps and stats of an analysed video
  GET  /api/tracks/{t}/runs/{stem}/frames   per-frame tracks + detections (compact, for overlays)
  GET  /api/tracks/{t}/marks/{stem}         saved manual marks;  POST saves them
  POST /api/tracks/{t}/manual               {stems} → learn the track from the saved marks
  GET  /api/video/{t}/{folder}/{name}       the video file (range requests)
  GET  /api/img?p=…                         an image inside the data folder
  POST /api/jobs                            {kind: analyse | learn, track, video?, gates?}
  GET  /api/jobs                            all jobs (status, progress, last log lines)
"""

import json
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
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset_paths import TrackPaths, data_root, list_tracks, under_root  # noqa: E402

STATIC = Path(__file__).parent / "static"
DET_MODEL = ROOT / "cyn_current-20260715_best.pt"     # the detector the pass scorer was trained with
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
FOLDERS = {"learn": "learn_videos", "race": "test_videos"}
ROLE_OF = {"learn_videos": "learn", "test_videos": "race", "memory_videos": "memory", "videos": "race"}

app = FastAPI(title="FPV spotter")
app.add_middleware(GZipMiddleware, minimum_size=2000)


# ──────────────────────────────────────────────────────────────
# Tracks and videos
# ──────────────────────────────────────────────────────────────

def _track(name: str) -> TrackPaths:
    tp = next((t for t in list_tracks() if t.dir.name == name), None)
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
    out = []
    for folder in ("learn_videos", "test_videos", "memory_videos", "videos"):
        d = tp.dir / folder
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            key = f"{f}:{f.stat().st_mtime}"
            if key not in _meta_cache:
                _meta_cache[key] = _video_meta(f)
            stem = f.stem
            out.append({
                "name": f.name, "stem": stem, "folder": folder, "role": ROLE_OF.get(folder, "race"),
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
            "learned_from": d.get("learned_from", []), "lap_seconds": lm.get("lap_seconds"),
            "extras": sum(p["kind"] == "extra" for p in pts)}


@app.get("/api/tracks")
def tracks():
    return {"data_root": str(data_root()),
            "tracks": [{"name": tp.dir.name, "videos": _videos(tp), "learned": _learned(tp)}
                       for tp in list_tracks()]}


@app.post("/api/tracks")
async def create_track(req: Request):
    name = str((await req.json()).get("name", "")).strip()
    if not re.fullmatch(r"[\w\- ]{1,64}", name) or name == "training":
        raise HTTPException(400, "Use letters, digits, spaces, - or _ for the track name.")
    tp = TrackPaths(data_root() / name).ensure()
    for f in FOLDERS.values():
        (tp.dir / f).mkdir(exist_ok=True)
    return {"name": name}


@app.post("/api/tracks/{track}/upload")
async def upload(track: str, role: str, file: UploadFile):
    tp = _track(track)
    if role not in FOLDERS:
        raise HTTPException(400, "role must be learn or race")
    name = Path(file.filename or "video.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "not a video file")
    dest = tp.dir / FOLDERS[role] / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while chunk := await file.read(8 << 20):
            f.write(chunk)
    return {"name": name, "folder": FOLDERS[role]}


@app.post("/api/tracks/{track}/import")
async def import_video(track: str, req: Request):
    """Copy a video that is already on this computer (no upload)."""
    tp = _track(track)
    body = await req.json()
    src = Path(str(body.get("path", ""))).expanduser()
    role = body.get("role", "race")
    if role not in FOLDERS:
        raise HTTPException(400, "role must be learn or race")
    if not src.is_file() or src.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, f"not a video file: {src}")
    dest = tp.dir / FOLDERS[role] / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    return {"name": src.name, "folder": FOLDERS[role]}


@app.get("/api/browse")
def browse(dir: str = "~/Downloads"):
    """Folders and video files in a folder on this computer (for the Add video dialog)."""
    d = Path(dir).expanduser()
    if not d.is_dir():
        d = Path.home()
    d = d.resolve()
    dirs, vids = [], []
    try:
        entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        raise HTTPException(403, f"No permission to read {d}")
    for p in entries:
        if p.name.startswith("."):
            continue
        try:
            if p.is_dir():
                dirs.append(p.name)
            elif p.suffix.lower() in VIDEO_EXT:
                st = p.stat()
                vids.append({"name": p.name, "path": str(p), "mb": round(st.st_size / 1e6, 1),
                             "mtime": st.st_mtime})
        except OSError:
            continue
    home = Path.home()
    places = [("Downloads", home / "Downloads"), ("Desktop", home / "Desktop"), ("Movies", home / "Movies"),
              ("Documents", home / "Documents"), ("Home", home), ("Volumes", Path("/Volumes"))]
    return {"dir": str(d), "parent": str(d.parent) if d != d.parent else None, "dirs": dirs, "videos": vids,
            "places": [{"name": n, "path": str(p)} for n, p in places if p.is_dir()]}


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
    if under_root(f) is None or not f.is_file():
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
               "reason": p.get("reason", ""), "img": _img_url(p.get("query_img", ""))}
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
    tp = _track(track)
    marks = (await req.json()).get("marks", [])
    tp.runs_dir.mkdir(parents=True, exist_ok=True)
    (tp.runs_dir / f"{stem}.marks.json").write_text(json.dumps({"marks": marks}, indent=1), encoding="utf-8")
    return {"saved": len(marks)}


@app.post("/api/tracks/{track}/manual")
async def learn_manual(track: str, req: Request):
    """Learn the track from the marks saved on one or more analysed videos."""
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
        cmd = py + [str(ROOT / "extract_race.py"), "--video", str(tp.dir / v["folder"] / v["name"]),
                    "--det-model", str(DET_MODEL), "--clip-device", "mps" if _has_mps() else "cpu"]
        title, after = f"Analyse {v['name']}", None
    elif kind == "learn":
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
    return {"jobs": sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)}


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
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
