#!/usr/bin/env python3
"""
perf_run.py — time a full race analysis (extract_race.py, as the web app starts it), by stage.

Runs on a throw-away copy of the video's track (its learned gates and timing, the video linked in), so the
track's own results are never touched. Prints one JSON line per run: frames, the frame loop's speed (YOLO +
tracker + motion + CLIP at passes), the post-processing (pass scorer, scene embeddings, near-misses, gate
labelling), the total and the speed against real time, plus the passes and laps it found.

    python perf_run.py "<FPVdatasets>/fril-26-race-7/test_videos/Q8 Niv .mp4" --runs 2
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_once(video: Path, det_model: str, clip_device: str) -> dict:
    tp_dir = video.parent.parent
    with tempfile.TemporaryDirectory(prefix="perf_") as tmp:
        data = Path(tmp)
        (data / "training").symlink_to(tp_dir.parent / "training")
        track = data / tp_dir.name
        (track / "test_videos").mkdir(parents=True)
        (track / "runs").mkdir()
        for f in ("gate_memory.json", "gate_timing.learned.json"):
            if (tp_dir / f).exists():
                shutil.copy2(tp_dir / f, track / f)
        (track / "test_videos" / video.name).symlink_to(video)
        cmd = [sys.executable, "-u", str(ROOT / "extract_race.py"), "--video", str(track / "test_videos" / video.name),
               "--det-model", det_model, "--clip-device", clip_device]
        t0, marks, frames, fps, done = time.time(), {}, 0, 30.0, ""
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**os.environ, "FPV_DATA_ROOT": str(data), "PYTHONIOENCODING": "utf-8"})
        for line in p.stdout:
            t = time.time() - t0
            if m := re.match(r"Video: (\d+) frames\s+([\d.]+) fps", line):
                frames, fps = int(m[1]), float(m[2])
            elif line.startswith("Running race analysis"):
                marks["loop_start"] = t
            elif line.startswith("Progress:"):
                marks["loop_end"] = t
            elif line.startswith("Done."):
                done = line.split("(")[0].replace("Done.", "").strip()
        if p.wait() != 0 or "loop_end" not in marks:
            raise SystemExit(f"extract_race failed on {video.name}")
    total = time.time() - t0
    loop = marks["loop_end"] - marks["loop_start"]
    return {"video": video.name, "frames": frames, "video_s": round(frames / fps, 1),
            "startup_s": round(marks["loop_start"], 1), "frame_loop_s": round(loop, 1),
            "frame_loop_fps": round(frames / loop, 1), "post_s": round(total - marks["loop_end"], 1),
            "total_s": round(total, 1), "x_realtime": round(frames / fps / total, 2), "found": done}


def main():
    ap = argparse.ArgumentParser(description="Time a full race analysis by stage")
    ap.add_argument("video", type=Path, help="a video in <track>/test_videos/")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--det-model", default=str(ROOT / "cyn_current-20260715_best.pt"))
    ap.add_argument("--clip-device", default=None, help="cpu / mps / cuda (default: as the web app picks)")
    args = ap.parse_args()
    if args.clip_device is None:
        import torch
        args.clip_device = "mps" if torch.backends.mps.is_available() else "cpu"
    for _ in range(args.runs):
        print(json.dumps(run_once(args.video.resolve(), args.det_model, args.clip_device)), flush=True)


if __name__ == "__main__":
    main()
