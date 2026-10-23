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
  - a stream URL: rtsp://… (e.g. the media box, rtsp://<box-address>:8554/<path>) or http://… (MJPEG);
  - a stream sent to this machine (--listen): it waits for the sender (e.g. the media box streaming out to
    rtmp://<this machine>:1935/live/feed) and again after the sender drops off, until stopped.

    python online_spotter.py "<FPVdatasets>/fril-27-first-track/test_videos/h007_R3_Storm.mp4" --compare
    python online_spotter.py <video> --track fril-27-first-track     # + gate names and laps
    python online_spotter.py 0                 # capture device 0
    python online_spotter.py rtsp://<box-address>:8554/<path>
    python online_spotter.py rtmp://0.0.0.0:1935/live/feed --listen      # the box sends to this machine

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
IMGSZ = 320                 # detector input: 640 is the offline size; 320 loses nothing to it (live_bench.py) at 1/3 the CPU
STATS_EVERY = 0.5           # seconds between "stats" events
CLIP_DEVICE = "cpu"         # CLIP on the GPU next to YOLO makes the detector drop frames (4% vs 0.8%)
DEVICE_FPS = 30             # asked of a capture device
LISTEN_WAIT_MS = 2000       # a stream sent to us: each wait for the sender (then again, until stopped)
EMBED_BACK = 4              # the gate crop is taken 4 frames before the pass fires, as offline (extract_race.py)
def frame_step_kwargs(tracker_kw: dict, passdet_kw: dict, k: int):
    """The tracker and pass detector are tuned for every frame at 30 fps; with every k-th frame, time-outs grow k
    times, frame counts shrink k times, and per-frame smoothing (EMA) becomes its k-frame equivalent. (Keeping 3
    aligned frames at 15 fps — 0.2 s — misses start gates that are in view only briefly.)"""
    if k <= 1:
        return tracker_kw, passdet_kw
    ema = lambda a: 1.0 - (1.0 - a) ** k
    t = {**tracker_kw, "ttl_seconds": tracker_kw["ttl_seconds"] * k, "ema_alpha": ema(tracker_kw["ema_alpha"])}
    p = {**passdet_kw, "disappear_timeout": passdet_kw["disappear_timeout"] * k,
         "min_aligned_frames": max(1, -(-passdet_kw["min_aligned_frames"] // k)),
         "area_vel_ema_alpha": ema(passdet_kw["area_vel_ema_alpha"]),
         "approach_score_alpha": ema(passdet_kw["approach_score_alpha"])}
    return t, p


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

    def __init__(self, spec: Spec, realtime: bool = True, speed: float = 1.0, stride: int = 1,
                 on_raw: Optional[Callable[[np.ndarray, float], None]] = None, listen: bool = False,
                 stop: Optional[threading.Event] = None):
        """stride: hand on only every stride-th frame (a slower spotter, e.g. 15 fps from a 30 fps feed).
        on_raw(frame, now): every frame the feed delivers, before any is skipped or dropped (feed_recorder).
        listen: spec is an rtmp:// or rtsp:// address of this machine the sender streams to; waits for it (until
        stop is set: then .cap is None)."""
        self.on_raw = on_raw
        self.spec, self.realtime, self.speed, self.stride = spec, realtime, max(0.1, speed), max(1, int(stride))
        self.kind = "device" if isinstance(spec, int) else "listen" if listen else "url" if "://" in str(spec) else "file"
        self._stop = threading.Event()
        self._ext_stop = stop or self._stop               # the caller's: stops a wait for the sender too
        if self.kind == "listen":
            self.cap = self._listen()
            if self.cap is None:
                self.fps, self.frames, self.read, self.dropped, self.ended = 30.0, 0, 0, 0, True
                return
        else:
            if self.kind == "url" and str(spec).startswith("rtsp"):
                # small buffers: a frame should be handed on as soon as it's decoded
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            if self.kind == "device" and platform.system() == "Darwin":
                self.cap = cv2.VideoCapture(spec, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(spec)
            if self.kind == "device" and self.cap.isOpened():
                # left alone, AVFoundation picks a capture card's slowest mode (a USB3 HDMI card: 20 fps at 1080p);
                # 30 is what the spotter and the learned timings are made for
                self.cap.set(cv2.CAP_PROP_FPS, DEVICE_FPS)
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
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _listen(self):
        """Wait for the sender to connect (FFmpeg as the server, one sender); None if stopped first."""
        url = str(self.spec)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            ("rtsp_flags;listen" if url.startswith("rtsp") else "listen;1")
            + "|analyzeduration;500000|probesize;500000|fflags;nobuffer|flags;low_delay")
        quick, stopped = 0, lambda: self._stop.is_set() or self._ext_stop.is_set()
        while not stopped():
            t = time.monotonic()
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, LISTEN_WAIT_MS])
            if cap.isOpened():
                return cap
            cap.release()
            if time.monotonic() - t < LISTEN_WAIT_MS / 4000:   # it didn't even wait: the port is taken
                quick += 1
                if quick >= 3:
                    raise RuntimeError(f"Can't receive on port {url.split('/')[2].split(':')[-1]}: is another program using it?")
                self._stop.wait(0.5)
            else:
                quick = 0
        return None

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def restart_clock(self):
        """A live feed's time and counts from now (a preview handed over to a run: its passes from 0 s)."""
        self.t0, self.read, self.dropped = time.monotonic(), 0, 0

    def _run(self):
        idx, fails = 0, 0
        self.t0 = t0 = time.monotonic()
        while not self._stop.is_set():
            if self.kind == "file" and idx % self.stride:     # a frame that won't be analysed: skip, don't convert
                if not self.cap.grab():
                    break
                idx += 1
                continue
            ok, frame = self.cap.read()
            if not ok:
                if self.kind == "file":
                    break
                if self.kind == "listen":                # the sender dropped off: wait for it again
                    self.cap.release()
                    try:
                        self.cap = self._listen()
                    except RuntimeError:
                        self.cap = None
                    if self.cap is None:
                        break
                    continue
                fails += 1                       # a live feed hiccups; give up after ~5 s without a frame
                if fails > 250:
                    break
                time.sleep(0.02)
                continue
            fails = 0
            now = time.monotonic()
            if self.on_raw is not None:
                try:
                    self.on_raw(frame, now)
                except Exception as e:                   # noqa: BLE001 — recording must not stop the spotter
                    print(f"[feed] on_raw: {e}", file=sys.stderr)
                    self.on_raw = None
            if self.kind == "file":
                t = idx / self.fps
                if self.realtime:
                    due = t0 + t / self.speed
                    if due > now:
                        self._stop.wait(due - now)
                    now = max(now, due)
            else:
                t = now - self.t0
            idx += 1
            if self.kind != "file" and (idx - 1) % self.stride:
                continue
            with self._cond:
                if not self.realtime:            # --fast: every frame is processed, none dropped
                    while self._new is not None and not self._stop.is_set():
                        self._cond.wait(0.1)
                if self._new is not None:
                    self.dropped += 1
                self._new = (frame, t, now)
                self.read += 1
                self._cond.notify_all()
        if self.cap is not None:
            self.cap.release()
        with self._cond:
            self.ended = True
            self._cond.notify_all()

    def get(self, timeout: Optional[float] = None):
        """The newest frame not yet taken, waiting for one; None once the feed has ended or was stopped (or after
        timeout seconds without a frame)."""
        until = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._new is None and not self.ended and not self._stop.is_set():
                if until is not None and time.monotonic() >= until:
                    return None
                self._cond.wait(0.2)
            item, self._new = self._new, None
            self._cond.notify_all()
            return item


class OnlineSpotter:
    """The detector stays loaded between runs; each run() gets a fresh tracker and pass detector."""

    def __init__(self, det_model: Union[str, Path] = DET_MODEL, imgsz: int = IMGSZ, conf: float = 0.25,
                 device: Optional[str] = None, detector=None, clip=None):
        """detector / clip: other backends (box_runtime.NcnnDetector / OrtClip on the media box, no PyTorch);
        by default the Ultralytics model det_model and open_clip."""
        self.imgsz, self.conf = int(imgsz), float(conf)
        self._clip = clip
        self.device = device or "cpu"
        if detector is not None:
            self.names, self.det_imgsz = detector.names, detector.hw
            self.detect = lambda frame: self._boxes(detector(frame, self.conf))
            self.detect(np.zeros((540, 960, 3), np.uint8))                     # warm-up
            return
        import torch
        from ultralytics import YOLO
        from lazy_spotter import _get_yolo_names
        self.det = YOLO(str(det_model))
        if device is None and not str(det_model).endswith(".mlpackage"):
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.kw = {"device": device} if device else {}
        # an exported model (ONNX / NCNN) has a fixed input: a 16:9 feed letterboxed to imgsz wide (320 → 320×192)
        self.det_imgsz = self.imgsz if str(det_model).endswith(".pt") else \
            (int(np.ceil(self.imgsz * 9 / 16 / 32) * 32), self.imgsz)
        self.names = _get_yolo_names(self.det)
        self.det(np.zeros((540, 960, 3), np.uint8), imgsz=self.det_imgsz, verbose=False, **self.kw)   # warm-up
        self.device = device or "cpu"

    def detect(self, frame: np.ndarray):
        """(boxes as int xyxy, confidences, classes) of one frame."""
        res = self.det(frame, imgsz=self.det_imgsz, conf=self.conf, verbose=False, max_det=50, **self.kw)[0]
        if not len(res.boxes):
            return np.zeros((0, 4), int), np.zeros(0), np.zeros(0, int)
        return (res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int))

    @staticmethod
    def _boxes(dets):
        if not dets:
            return np.zeros((0, 4), int), np.zeros(0), np.zeros(0, int)
        a = np.array(dets, dtype=float)
        return a[:, :4].astype(int), a[:, 4], a[:, 5].astype(int)

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
            stop: Optional[threading.Event] = None, decoder=None, frame_step: Optional[int] = None,
            keep_source: bool = False) -> dict:
        """decoder: an online_decoder.OnlineDecoder (stage 2) — each pass is then followed by a "gate" event
        naming it, and "start" / "lap" events. keep_source: the feed stays open after the run (the web app's
        preview goes on with it)."""
        import pipeline_cfg
        from track_core import TimeTracker, clamp_bbox, _cls_to_name, _crop_padded
        from pass_detector import PassDetector, detect_camera_edges
        cfg = pipeline_cfg.load()
        step = source.stride if frame_step is None else frame_step
        tkw, pkw = frame_step_kwargs(pipeline_cfg.tracker_kwargs(cfg), pipeline_cfg.gates_passdet_kwargs(cfg), step)
        tracker = TimeTracker(**tkw)
        passdet = PassDetector(**pkw)
        stop = stop or threading.Event()

        n_pass, done, lags, last_pass = 0, 0, [], None
        w_t0, w_done, w_read, w_drop = time.monotonic(), 0, source.read, source.dropped
        last_stats = w_t0
        stats = {}
        edges_set = False
        # (frame, {track id: bbox}) of the last frames: the gate crop is EMBED_BACK feed frames before the pass
        recent = deque(maxlen=max(1, round(EMBED_BACK / max(1, source.stride))) + 1)
        jobs: "queue.Queue" = queue.Queue()
        namer = None
        if decoder is not None:
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
            item = source.get(timeout=0.5)
            if item is None:
                if source.ended or source._stop.is_set():
                    break
                continue                                 # a live feed waiting for its sender
            frame, t, arrival = item
            H, W = frame.shape[:2]
            if not edges_set:
                passdet.set_camera_edges(*detect_camera_edges(frame))
                edges_set = True

            xyxy, confs, clss = self.detect(frame)
            typed = []
            if len(xyxy):
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
                # a gate that disappeared fires one analysed frame after it was last seen: time it from then plus one
                # feed frame, as at 30 fps, so the pass time doesn't depend on how many frames are analysed
                tp = min(t, float(evt.get("last_seen", t)) + 1.0 / source.fps)
                last_pass = {"type": "pass", "n": n_pass, "t": round(tp, 3), "track_id": int(evt.get("track_id", -1)),
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
                         "size": f"{W}×{H}", "imgsz": self.imgsz, "stride": source.stride, "cpu_temp": cpu_temp()}
                on_event(stats)
                w_t0, w_done, w_read, w_drop, last_stats = now, done, source.read, source.dropped, now
        if not keep_source:
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


def cpu_temp() -> Optional[float]:
    """The CPU temperature on a Raspberry Pi (Linux), else None."""
    try:
        return round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0, 1)
    except (OSError, ValueError):
        return None


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
    ap.add_argument("--listen", action="store_true", help="the source is an rtmp:// or rtsp:// address of this "
                    "machine: wait for the sender to stream to it")
    ap.add_argument("--fast", action="store_true", help="a file: every frame, as fast as possible (no drops)")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame (2: 15 fps from a 30 fps feed), "
                                                          "with the tracker / pass detector rescaled to it")
    ap.add_argument("--compare", action="store_true", help="a file: match the passes to its offline analysis")
    ap.add_argument("--det-model", default=str(DET_MODEL))
    ap.add_argument("--track", help="name gates and time laps with this track's learned gates (default: a file's "
                                    "own track, if learned); 'none' for stage 1 only")
    box = ap.add_argument_group("on the media box (box_remote.py runs it so; no PyTorch)")
    box.add_argument("--box-detector", help="the detector exported to ONNX: run it with NCNN (box_runtime.py)")
    box.add_argument("--box-clip", help="CLIP's image encoder exported to ONNX: run it with ONNX Runtime (int8)")
    box.add_argument("--track-dir", help="a track bundle: gate_memory.json + gate_timing.learned.json")
    box.add_argument("--jsonl", action="store_true", help="every event as a JSON line on stdout (the rest on stderr)")
    box.add_argument("--preview", type=int, default=0, help="with --jsonl: a JPEG frame every N analysed frames")
    box.add_argument("--exit-on-eof", action="store_true", help="stop when stdin closes (the SSH session ended)")
    args = ap.parse_args()

    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if args.jsonl else print
    emit_lock = threading.Lock()

    def emit(obj):
        with emit_lock:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()

    spec = parse_spec(args.source)
    stop = threading.Event()
    if args.exit_on_eof:
        def watch_stdin():
            sys.stdin.read()                 # returns when the controlling session goes away
            stop.set()
            time.sleep(3)
            os._exit(0)
        threading.Thread(target=watch_stdin, daemon=True).start()
    if args.box_detector:
        import box_runtime
        if args.jsonl:
            emit({"type": "status", "detail": "preparing the models on the box…"})
        hw = (int(np.ceil(args.imgsz * 9 / 16 / 32) * 32), args.imgsz)
        param, clip_q = box_runtime.prepare(Path(args.box_detector).parent, args.box_detector, args.box_clip, hw, log)
        det = box_runtime.NcnnDetector(param, box_runtime.detector_names(args.box_detector), hw)
        sp = OnlineSpotter(imgsz=args.imgsz, conf=args.conf, detector=det,
                           clip=box_runtime.OrtClip(clip_q) if args.box_clip else None)
    else:
        log(f"Loading detector ({Path(args.det_model).name}, input {args.imgsz})…")
        sp = OnlineSpotter(args.det_model, args.imgsz, args.conf)
    if args.listen:
        log(f"Waiting for the stream at {spec} …")
    src = FrameSource(spec, realtime=not args.fast, speed=args.speed, stride=args.stride, listen=args.listen)
    log(f"Feed: {describe(spec)}  {src.fps:.0f} fps" + (f", {src.frames / src.fps:.0f} s" if src.frames else ""))
    decoder = None
    if args.track_dir:
        import online_decoder
        td = Path(args.track_dir)
        decoder = online_decoder.OnlineDecoder(td / "gate_memory.json", td / "gate_timing.learned.json")
        log(f"Track bundle {td.name}: {decoder.n} gates, timing {decoder.timing or 'none (order + appearance only)'}")
    elif args.track != "none":
        import online_decoder
        from dataset_paths import list_tracks, track_of
        tp = (next((t for t in list_tracks() if t.dir.name == args.track), None) if args.track
              else track_of(Path(args.source).resolve()) if src.kind == "file" else None)
        if args.track and tp is None:
            sys.exit(f"No track {args.track}")
        if tp is not None and tp.gate_memory.exists():
            decoder = online_decoder.for_track(tp)
            log(f"Track: {tp.name}, {decoder.n} gates, timing {decoder.timing or 'none (order + appearance only)'}")
    got = []
    if args.jsonl:
        emit({"type": "ready", "fps": src.fps, "frames": src.frames, "gates": decoder.n if decoder else None,
              "timing": decoder.timing if decoder else None, "stride": src.stride, "imgsz": args.imgsz,
              "host": platform.node()})

    def on_event(e):
        if args.jsonl:
            if e["type"] == "pass":
                got.append(e)
            emit(e)
            return
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

    on_frame = None
    if args.jsonl and args.preview > 0:
        import base64
        count = [0]

        def on_frame(frame, t, tracks, states, last_pass):
            count[0] += 1
            if count[0] % args.preview:
                return
            img = draw(frame, tracks, states, last_pass, t)
            img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                emit({"type": "frame", "t": round(t, 3), "jpg": base64.b64encode(buf.tobytes()).decode()})

    try:
        sp.run(src.start(), on_event, on_frame, stop, decoder=decoder)
    except KeyboardInterrupt:
        src.stop()
    if args.compare and src.kind == "file":
        compare(Path(args.source).resolve(), got)


if __name__ == "__main__":
    main()
