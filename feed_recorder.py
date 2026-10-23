#!/usr/bin/env python3
"""
feed_recorder.py — save a live feed to an MP4 while the live spotter runs on it.

At a new track there is nothing learned yet: the Live tab runs on the pilots' feeds with gate-pass beeps only and
saves each feed, to learn the track from later (Learn · auto). Every frame the feed delivers is saved at full size
— the spotter may skip or drop frames, the recording doesn't — on the wall clock: a frame is repeated or left out
so the video's length is the real time, since learning uses the time between gates. The feed's first second is
only timed (its real frame rate), not saved.

Encoded by ffmpeg (VideoToolbox H.264 on a Mac, libx264 otherwise) in its own process, fed from a queue, so the
spotter isn't held up.

    rec = FeedRecorder(path, fps); rec.write(frame, now) …; info = rec.close()
"""

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

QUEUE_FRAMES = 90                     # ~3 s of frames waiting for the encoder at most
PROBE_S = 1.0                         # the feed's real frame rate is measured over its first second
COMMON_FPS = (24, 25, 30, 48, 50, 60, 100, 120)
BITS_PER_PIXEL = 0.15                 # → ~2.3 Mbit/s at 960×540 30 fps, ~9 Mbit/s at 1080p30


def _encoder(w: int, h: int, fps: float) -> list:
    kbps = max(1500, int(w * h * fps * BITS_PER_PIXEL / 1000))
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    if "h264_videotoolbox" in out:
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{kbps}k", "-realtime", "1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


class FeedRecorder:
    def __init__(self, path: Path, fps: float):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Saving the feed needs ffmpeg (brew install ffmpeg).")
        self.path, self.fps = Path(path), float(fps if 1 <= fps <= 240 else 30.0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: Optional[subprocess.Popen] = None
        self.q: "queue.Queue" = queue.Queue(maxsize=QUEUE_FRAMES)
        self.t0: Optional[float] = None   # when the file's first frame arrived
        self.written = 0                  # frames in the file (repeats included)
        self.received = 0
        self.size = None
        self.error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._probe: list = []            # arrival times of the first second, before the encoder starts

    def _measure_fps(self) -> float:
        """The rate frames really arrive at; the reported one if it agrees (OpenCV says 25 for MJPEG streams)."""
        ts = self._probe
        if len(ts) < 5 or ts[-1] <= ts[0]:
            return self.fps
        real = (len(ts) - 1) / (ts[-1] - ts[0])
        if abs(real - self.fps) <= 0.1 * self.fps:
            return self.fps
        return float(min(COMMON_FPS, key=lambda f: abs(f - real)))

    def _start(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.fps = self._measure_fps()
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{self.fps:.3f}", "-i", "-", *_encoder(w, h, self.fps), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(self.path.with_suffix(".part.mp4"))]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self.proc.stdin.write(item)
            except (BrokenPipeError, OSError) as e:
                self.error = f"the encoder stopped: {e}"
                return

    def write(self, frame: np.ndarray, now: Optional[float] = None):
        """One frame of the feed, arriving at now (monotonic seconds)."""
        if self.error:
            return
        now = time.monotonic() if now is None else now
        if self.proc is None:
            self._probe.append(now)
            if now - self._probe[0] < PROBE_S:
                return
            self._start(frame)
            self._probe = []
            self.t0 = now
        self._add(frame, now)

    def _add(self, frame: np.ndarray, now: float):
        if frame.shape[1::-1] != self.size:          # the feed changed size: keep the file's size
            import cv2
            frame = cv2.resize(frame, self.size)
        self.received += 1
        due = int(round((now - self.t0) * self.fps)) + 1   # frames the file should hold by now
        n = due - self.written
        if n <= 0:                                    # ahead of the clock: leave this one out
            return
        data = np.ascontiguousarray(frame).tobytes()
        for _ in range(min(n, int(self.fps))):        # behind: repeat it (a stalled feed: at most a second)
            self.q.put(data)
            self.written += 1

    def info(self) -> dict:
        return {"file": self.path.name, "seconds": round(self.written / self.fps, 1), "frames": self.written,
                "size": f"{self.size[0]}×{self.size[1]}" if self.size else None, "error": self.error}

    def close(self) -> dict:
        """Finish the file (→ path, without the .part); returns info()."""
        if self.proc is None:
            self.error = self.error or "stopped before anything was saved (the first second times the feed)"
        if self.proc is not None:
            self.q.put(None)
            if self._thread:
                self._thread.join(timeout=30)
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.error = self.error or "the encoder didn't finish"
            part = self.path.with_suffix(".part.mp4")
            if self.proc.returncode == 0 and part.exists():
                part.replace(self.path)
            elif not self.error:
                self.error = (self.proc.stderr.read().decode(errors="replace")[-300:] or "encoding failed")
        return self.info()
