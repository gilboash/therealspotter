#!/usr/bin/env python3
"""
online_spotter.py — the real-time spotter: one feed in, gate passes out while the drone flies.

Stage 1: YOLO + tracker + the live PassDetector; every pass is an event the moment the detector fires.
Stage 2 (with a learned track): each pass is named (G1…GN) by online_decoder.py — order + the track's learned
timing + CLIP of the gate — and laps run G1 to G1, the first G1 pass starting lap 1. CLIP and the decoder run in a
second thread, so the detector never waits for them. The web app's Live tab plays a tone per gate and calls the
lap times out.

A feed behaves like a live one whatever it is: a reader thread keeps only the newest frame, and frames the
spotter could not take in time are dropped (and counted), so it never falls behind real time.
  - a video file is played at its own pace (--speed scales it); --fast instead processes every frame as fast
    as possible, like the offline analysis (to compare the two);
  - a capture device (USB / HDMI card) by index;
  - a stream URL: rtsp://… (e.g. the media box, rtsp://naco.local:8554/<path>) or http://… (MJPEG).

    python online_spotter.py "<FPVdatasets>/fril-27-first-track/test_videos/h007_R3_Storm.mp4" --compare
    python online_spotter.py <video> --track fril-27-first-track     # + gate names and laps
    python online_spotter.py 0                 # capture device 0
    python online_spotter.py rtsp://naco.local:8554/live

--compare matches the passes against the video's offline analysis (<track>/runs/<video>.race_data.json).
"""

import argparse
import json
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DET_MODEL = ROOT / "cyn_current-20260715_best.pt"
IMGSZ = 416                 # detector input: 640 is the offline size; 416 keeps a single feed at ~40 fps on an M2
STATS_EVERY = 0.5           # seconds between "stats" events
CLIP_DEVICE = "cpu"         # CLIP on the GPU next to YOLO makes the detector drop frames (4% vs 0.8%)
EMBED_BACK = 4              # the gate crop is taken 4 frames before the pass fires, as offline (extract_race.py)
STAGE_BGR = {"idle": (173, 163, 154), "aligned": (61, 200, 255), "passed": (123, 217, 76)}

Spec = Union[str, int]


def parse_spec(s: str) -> Spec:
    """A device index ("0") or a path / URL."""
    return int(s) if s.strip().isdigit() else s


def describe(spec: Spec) -> str:
    if isinstance(spec, int):
        return f"capture device {spec}"
    return spec if "://" in spec else Path(spec).name


class FrameSource:
    """One feed. get() returns the newest frame as (frame, t, arrival): t is the frame's time in the feed
    (seconds), arrival the monotonic clock when it became available."""

    def __init__(self, spec: Spec, realtime: bool = True, speed: float = 1.0):
        self.spec, self.realtime, self.speed = spec, realtime, max(0.1, speed)
        self.kind = "device" if isinstance(spec, int) else "url" if "://" in str(spec) else "file"
        if self.kind == "url" and str(spec).startswith("rtsp"):
            # small buffers: a frame should be handed on as soon as it's decoded
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                                  "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0")
        if self.kind == "device" and platform.system() == "Darwin":
            self.cap = cv2.VideoCapture(spec, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(spec)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {describe(spec)}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not 1 <= self.fps <= 240:
            self.fps = 30.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.kind == "file" else 0
        self.read = self.dropped = 0
        self.ended = False
        self._new = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _run(self):
        t0, idx, fails = time.monotonic(), 0, 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.kind == "file":
                    break
                fails += 1                       # a live feed hiccups; give up after ~5 s without a frame
                if fails > 250:
                    break
                time.sleep(0.02)
                continue
            fails = 0
            now = time.monotonic()
            if self.kind == "file":
                t = idx / self.fps
                if self.realtime:
                    due = t0 + t / self.speed
                    if due > now:
                        self._stop.wait(due - now)
                    now = max(now, due)
            else:
                t = now - t0
            idx += 1
            with self._cond:
                if not self.realtime:            # --fast: every frame is processed, none dropped
                    while self._new is not None and not self._stop.is_set():
                        self._cond.wait(0.1)
                if self._new is not None:
                    self.dropped += 1
                self._new = (frame, t, now)
                self.read += 1
                self._cond.notify_all()
        self.cap.release()
        with self._cond:
            self.ended = True
            self._cond.notify_all()

    def get(self):
        """The newest frame not yet taken, waiting for one; None once the feed has ended or was stopped."""
        with self._cond:
            while self._new is None and not self.ended and not self._stop.is_set():
                self._cond.wait(0.2)
            item, self._new = self._new, None
            self._cond.notify_all()
            return item


class OnlineSpotter:
    """The detector stays loaded between runs; each run() gets a fresh tracker and pass detector."""

    def __init__(self, det_model: Union[str, Path] = DET_MODEL, imgsz: int = IMGSZ, conf: float = 0.25,
                 device: Optional[str] = None):
        import torch
        from ultralytics import YOLO
        from lazy_spotter import _get_yolo_names
        self.det = YOLO(str(det_model))
        if device is None and not str(det_model).endswith(".mlpackage"):
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.kw = {"device": device} if device else {}
        self.imgsz, self.conf = int(imgsz), float(conf)
        self.names = _get_yolo_names(self.det)
        self.det(np.zeros((540, 960, 3), np.uint8), imgsz=self.imgsz, verbose=False, **self.kw)   # warm-up
        self.device = device or "cpu"
        self._clip = None

    @property
    def clip(self):
        """CLIP for the gate crops (stage 2), loaded the first time a run names gates."""
        if self._clip is None:
            from lazy_spotter import ClipEmbedder
            self._clip = ClipEmbedder(device=CLIP_DEVICE)
            self._clip.embed_bgr(np.zeros((224, 224, 3), np.uint8))
        return self._clip

    def run(self, source: FrameSource, on_event: Callable[[dict], None],
            on_frame: Optional[Callable[[np.ndarray, float, list, dict, Optional[dict]], None]] = None,
            stop: Optional[threading.Event] = None, decoder=None) -> dict:
        """decoder: an online_decoder.OnlineDecoder (stage 2) — each pass is then followed by a "gate" event
        naming it, and "start" / "lap" events."""
        import pipeline_cfg
        from lazy_spotter import TimeTracker, clamp_bbox, _cls_to_name
        from pass_detector import PassDetector, detect_camera_edges
        cfg = pipeline_cfg.load()
        tracker = TimeTracker(**pipeline_cfg.tracker_kwargs(cfg))
        passdet = PassDetector(**pipeline_cfg.gates_passdet_kwargs(cfg))
        stop = stop or threading.Event()

        n_pass, done, lags, last_pass = 0, 0, [], None
        w_t0, w_done, w_read, w_drop = time.monotonic(), 0, source.read, source.dropped
        last_stats = w_t0
        stats = {}
        edges_set = False
        recent = deque(maxlen=EMBED_BACK + 1)            # (frame, {track id: bbox}) of the last frames
        jobs: "queue.Queue" = queue.Queue()
        namer = None
        if decoder is not None:
            from extract_race import _crop_padded
            decoder.reset()
            clip = self.clip

            def name_passes():                           # CLIP + decoder, off the detector's thread
                while True:
                    job = jobs.get()
                    if job is None:
                        return
                    p, crop, arrival = job
                    emb = clip.embed_bgr(crop) if crop is not None and crop.size else None
                    for e in decoder.add(p["t"], emb, p["gate_type"]):
                        if e["type"] == "gate":
                            e["n"] = p["n"]
                            p["gate_id"] = e["gate_id"]
                            e["delay_ms"] = round(1000 * (time.monotonic() - arrival))
                        on_event(e)

            namer = threading.Thread(target=name_passes, daemon=True)
            namer.start()
        while not stop.is_set():
            item = source.get()
            if item is None:
                break
            frame, t, arrival = item
            H, W = frame.shape[:2]
            if not edges_set:
                passdet.set_camera_edges(*detect_camera_edges(frame))
                edges_set = True

            res = self.det(frame, imgsz=self.imgsz, conf=self.conf, verbose=False, max_det=50, **self.kw)[0]
            typed = []
            if len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    c = float(c)
                    typed.append({"bbox": clamp_bbox((x1, y1, x2, y2), W, H), "det_conf": c,
                                  "type": _cls_to_name(k, self.names) if c >= 0.20 else "NONE", "type_score": c})
            typed = sorted(typed, key=lambda d: d["det_conf"], reverse=True)[:10]
            tracks = tracker.update(typed, t)
            passdet.update(tracks, t, frame_w=W, frame_h=H)
            recent.append((frame, {int(tr.track_id): list(tr.bbox) for tr in tracks}))

            now = time.monotonic()
            lag = now - arrival
            lags.append(lag)
            done += 1
            while True:
                evt = passdet.pop_any_passed()
                if evt is None:
                    break
                n_pass += 1
                last_pass = {"type": "pass", "n": n_pass, "t": round(t, 3), "track_id": int(evt.get("track_id", -1)),
                             "gate_type": str(evt.get("type", "UNKNOWN")), "reason": str(evt.get("reason", "")),
                             "delay_ms": round(lag * 1000), "named": namer is not None}
                on_event(last_pass)
                if namer is not None:
                    old, boxes = recent[0]
                    box = boxes.get(last_pass["track_id"])
                    crop = _crop_padded(old, box)[0] if box else None
                    jobs.put((last_pass, crop, arrival))
            if on_frame:
                on_frame(frame, t, tracks, getattr(passdet, "states", {}) or {}, last_pass)

            if now - last_stats >= STATS_EVERY:
                dt = now - w_t0
                stats = {"type": "stats", "t": round(t, 2), "duration": round(source.frames / source.fps, 1),
                         "fps_feed": round((source.read - w_read) / dt, 1), "fps_spotter": round((done - w_done) / dt, 1),
                         "dropped_pct": round(100 * (source.dropped - w_drop) / max(1, source.read - w_read), 1),
                         "delay_ms": round(1000 * float(np.median(lags[-60:]))), "passes": n_pass,
                         "size": f"{W}×{H}", "imgsz": self.imgsz}
                on_event(stats)
                w_t0, w_done, w_read, w_drop, last_stats = now, done, source.read, source.dropped, now
        source.stop()
        if namer is not None:
            jobs.put(None)
            namer.join(timeout=10)
        laps = decoder.laps if decoder is not None else []
        summary = {"type": "end", "passes": n_pass, "laps": len(laps),
                   "best_lap": min((l["dt"] for l in laps), default=None), "frames_read": source.read, "frames_done": done,
                   "dropped": source.dropped, "delay_ms_median": round(1000 * float(np.median(lags))) if lags else None,
                   "delay_ms_p95": round(1000 * float(np.percentile(lags, 95))) if lags else None,
                   "stopped": stop.is_set()}
        on_event(summary)
        return summary


def draw(frame: np.ndarray, tracks: list, states: dict, last_pass: Optional[dict], t: float) -> np.ndarray:
    """The feed with the tracked gates (gray idle, yellow aligned, green passed) and a flash after a pass."""
    out = frame.copy()
    for tr in tracks:
        st = states.get(int(tr.track_id))
        stage = str(getattr(st, "stage", "idle")) if st else "idle"
        x1, y1, x2, y2 = map(int, tr.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), STAGE_BGR.get(stage, (255, 255, 255)), 1 if stage == "idle" else 3)
    if last_pass and t - last_pass["t"] < 0.5:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), STAGE_BGR["passed"], 10)
        g = last_pass.get("gate_id")
        label = f"PASS {last_pass['n']}" if g is None else f"G{g}" if g else "?"
        cv2.putText(out, label, (24, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, label, (24, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, STAGE_BGR["passed"], 2, cv2.LINE_AA)
    return out


def compare(video: Path, got: list, window: float = 0.35):
    """Match the real-time passes to the offline analysis: its decided passes (scorer, near-misses left out) and
    the live detector's passes from the same run (the same logic at full size and every frame)."""
    from dataset_paths import track_of
    tp = track_of(video)
    f = tp.race_data(video.stem) if tp else None
    if not f or not f.exists():
        print(f"\nNo offline analysis of {video.name} to compare with.")
        return
    data = json.loads(f.read_text(encoding="utf-8"))
    refs = {"offline passes (scorer)": [p["t"] for p in data.get("passes", [])
                                        if p.get("source") != "NOPASS" and p.get("reason") != "near_miss"],
            "offline live detector": [p["t"] for p in data.get("live_passes", [])]}
    ts = sorted(p["t"] for p in got)
    print()
    for name, ref in refs.items():
        ref = sorted(ref)
        used, offs = set(), []
        for r in ref:
            best = min((j for j in range(len(ts)) if j not in used and abs(ts[j] - r) <= window),
                       key=lambda j: abs(ts[j] - r), default=None)
            if best is not None:
                used.add(best)
                offs.append(ts[best] - r)
        m = len(offs)
        print(f"vs {name:24s}: {len(ref):3d} there, {m:3d} found ({100 * m / max(1, len(ref)):.0f}%), "
              f"{len(ref) - m:3d} missed, {len(ts) - m:3d} extra; "
              f"time offset median {1000 * float(np.median(offs)) if offs else 0:+.0f} ms")


def main():
    ap = argparse.ArgumentParser(description="Real-time gate passes from one feed (stage 1: tone per pass)")
    ap.add_argument("source", help="video file, capture device index, or stream URL")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help=f"detector input size (default {IMGSZ}; offline uses 640)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=1.0, help="a file's playback speed (real time = 1)")
    ap.add_argument("--fast", action="store_true", help="a file: every frame, as fast as possible (no drops)")
    ap.add_argument("--compare", action="store_true", help="a file: match the passes to its offline analysis")
    ap.add_argument("--det-model", default=str(DET_MODEL))
    ap.add_argument("--track", help="name gates and time laps with this track's learned gates (default: a file's "
                                    "own track, if learned); 'none' for stage 1 only")
    args = ap.parse_args()

    spec = parse_spec(args.source)
    print(f"Loading detector ({Path(args.det_model).name}, input {args.imgsz})…")
    sp = OnlineSpotter(args.det_model, args.imgsz, args.conf)
    src = FrameSource(spec, realtime=not args.fast, speed=args.speed)
    print(f"Feed: {describe(spec)}  {src.fps:.0f} fps" + (f", {src.frames / src.fps:.0f} s" if src.frames else ""))
    decoder = None
    if args.track != "none":
        import online_decoder
        from dataset_paths import list_tracks, track_of
        tp = (next((t for t in list_tracks() if t.dir.name == args.track), None) if args.track
              else track_of(Path(args.source).resolve()) if src.kind == "file" else None)
        if args.track and tp is None:
            sys.exit(f"No track {args.track}")
        if tp is not None and tp.gate_memory.exists():
            decoder = online_decoder.for_track(tp)
            print(f"Track: {tp.name}, {decoder.n} gates, timing {decoder.timing or 'none (order + appearance only)'}")
    got = []

    def on_event(e):
        if e["type"] == "pass":
            got.append(e)
            print(f"PASS {e['n']:3d}  t={e['t']:7.2f}s  {e['gate_type']:9s} {e['reason']:24s} delay {e['delay_ms']:4d} ms")
        elif e["type"] == "gate":
            print(f"      → {'G' + str(e['gate_id']) if e['gate_id'] else 'not a pass'}"
                  f"  (conf {e['conf']}, speed ×{e['scale']}, delay {e['delay_ms']} ms)")
        elif e["type"] == "start":
            print(f"  ▶ lap {e['lap']} started at {e['t']:.2f}s{' (estimated)' if e['estimated'] else ''}")
        elif e["type"] == "lap":
            print(f"  ⏱ LAP {e['lap']}: {e['dt']:.2f}s{' (estimated)' if e['estimated'] else ''}")
        elif e["type"] == "stats":
            print(f"  [{e['t']:6.1f}s] feed {e['fps_feed']:4.1f} fps, spotter {e['fps_spotter']:4.1f} fps, "
                  f"dropped {e['dropped_pct']:4.1f}%, delay {e['delay_ms']} ms", file=sys.stderr)
        elif e["type"] == "end":
            print(f"\nDone: {e['passes']} passes, {e['laps']} laps (best {e['best_lap']}); {e['frames_done']}/{e['frames_read']} frames analysed "
                  f"({e['dropped']} dropped); delay median {e['delay_ms_median']} ms, 95% {e['delay_ms_p95']} ms")

    try:
        sp.run(src.start(), on_event, decoder=decoder)
    except KeyboardInterrupt:
        src.stop()
    if args.compare and src.kind == "file":
        compare(Path(args.source).resolve(), got)


if __name__ == "__main__":
    main()
