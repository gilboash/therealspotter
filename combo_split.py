#!/usr/bin/env python3
"""
combo_split.py — a combo (split / quad) view of several pilots' feeds → one feed per pilot.

On race day the video system often shows (and records) the feeds side by side: four 1024×576 tiles in a 2048×1152
frame, CH1 top left, CH2 top right, CH3 bottom left, CH4 bottom right. Nothing marks the tiles off; where two feeds
meet the picture jumps (they're unrelated), every frame, at the same place. So:

  - detect_layout(frames): the columns / rows where the picture jumps far more than its neighbourhood, across a
    dozen frames → a grid of tiles; each tile trimmed of its feed's own black border, and marked inactive if it
    doesn't change (a channel with no pilot: grey with "CHn" in it).
  - split_video(): one decode of the combo, one constant 30 fps MP4 per active tile (the pipeline is made for
    30 fps; these recordings are ~60 fps with uneven frame times) — each then a pilot's video like any other.
  - crop(frame, tile): one tile of a live combo frame (the Live tab: one pilot of a combo feed).

    python combo_split.py <combo.mp4>                 # → <combo>_CH1.mp4 … next to it
    python combo_split.py <combo.mp4> --detect-only   # the layout, as JSON
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

JUMP = 4.0             # a tile edge: the picture jumps this many times more than ~15 px around it
EDGE_MARGIN = 0.05     # no split this close to the frame's edge
MIN_TILE = 0.2         # a tile is at least this share of the frame's width / height
DARK = 28              # a feed's own black border: mean brightness below this …
MAX_TRIM = 0.06        # … trimmed up to this share of the tile on each side
STILL = 3.0            # a tile changing less than this (temporal std, grey levels) has no pilot
OUT_FPS = 30
BITS_PER_PIXEL = 0.3   # tiles kept for learning: ~5.3 Mbit/s at 1024×576 30 fps


def sample_frames(path: str, n: int = 12) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(total * 0.05, total * 0.95, n).astype(int) if total > n else range(max(total, n)):
        if total > n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def _splits(jump: np.ndarray, length: int) -> List[int]:
    """Positions (pixel index of the first pixel after the edge) where the jump stands out."""
    base = cv2.blur(jump.reshape(1, -1).astype(np.float32), (31, 1)).ravel()
    ratio = jump / (base + 1.0)
    lo, hi = int(length * EDGE_MARGIN), int(length * (1 - EDGE_MARGIN))
    cand = sorted((i for i in range(lo, min(hi, len(ratio))) if ratio[i] > JUMP), key=lambda i: -ratio[i])
    picked: List[int] = []
    for i in cand:                                       # strongest first, none within 2% of another
        if all(abs(i - j) > length * 0.02 for j in picked):
            picked.append(i)
    picked = sorted(p + 1 for p in picked)
    edges = [0] + picked + [length]
    if any(b - a < length * MIN_TILE for a, b in zip(edges, edges[1:])):
        return []                                        # not a grid of feeds
    return picked


def _trim(mean: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """The tile without its feed's own black border (columns / rows at its edges that are dark throughout)."""
    t = mean[y0:y1, x0:x1]
    h, w = t.shape
    cols, rows = t.mean(0), t.mean(1)
    l = 0
    while l < w * MAX_TRIM and cols[l] < DARK:
        l += 1
    r = w
    while w - r < w * MAX_TRIM and cols[r - 1] < DARK:
        r -= 1
    tp = 0
    while tp < h * MAX_TRIM and rows[tp] < DARK:
        tp += 1
    b = h
    while h - b < h * MAX_TRIM and rows[b - 1] < DARK:
        b -= 1
    x, y, ww, hh = x0 + l, y0 + tp, r - l, b - tp
    return x, y, ww - ww % 2, hh - hh % 2                # even sizes for H.264


def detect_layout(frames: List[np.ndarray]) -> dict:
    """{"size": [W, H], "grid": [cols, rows], "tiles": [{"ch", "x", "y", "w", "h", "active"}]}; a single tile
    (the whole frame) when it isn't a combo."""
    if not frames:
        raise ValueError("no frames")
    H, W = frames[0].shape[:2]
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    jx = np.mean([np.median(np.abs(np.diff(f, axis=1)), axis=0) for f in g], axis=0)
    jy = np.mean([np.median(np.abs(np.diff(f, axis=0)), axis=1) for f in g], axis=0)
    xs, ys = _splits(jx, W), _splits(jy, H)
    mean, std = g.mean(0), g.std(0)
    xe, ye = [0] + xs + [W], [0] + ys + [H]
    tiles = []
    for r in range(len(ye) - 1):
        for c in range(len(xe) - 1):
            x, y, w, h = _trim(mean, xe[c], ye[r], xe[c + 1], ye[r + 1])
            moving = float(np.median(std[y:y + h, x:x + w])) if len(frames) > 1 else STILL + 1
            tiles.append({"ch": len(tiles) + 1, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                          "active": moving > STILL, "motion": round(moving, 1)})
    return {"size": [W, H], "grid": [len(xe) - 1, len(ye) - 1], "tiles": tiles}


def is_combo(layout: dict) -> bool:
    return len(layout["tiles"]) > 1


def crop(frame: np.ndarray, tile: dict) -> np.ndarray:
    return frame[tile["y"]:tile["y"] + tile["h"], tile["x"]:tile["x"] + tile["w"]]


def _duration(path: str) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _encoder(w: int, h: int) -> list:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in out:
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{int(w * h * OUT_FPS * BITS_PER_PIXEL / 1000)}k"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


def split_video(path: str, out_dir: Optional[str] = None, layout: Optional[dict] = None,
                channels: Optional[List[int]] = None, log=print) -> List[Path]:
    """One MP4 per active tile (or the channels asked for): <stem>_CH<n>.mp4 in out_dir (default: next to it).
    Prints "Progress: x%" lines (the web app's jobs read them)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Splitting needs ffmpeg (brew install ffmpeg).")
    src = Path(path)
    layout = layout or detect_layout(sample_frames(str(src)))
    if not is_combo(layout):
        raise RuntimeError(f"{src.name} doesn't look like a combo view (no tile edges found).")
    tiles = [t for t in layout["tiles"] if (t["ch"] in channels if channels else t["active"])]
    if not tiles:
        raise RuntimeError("No channel with a pilot in it.")
    out = Path(out_dir) if out_dir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    dests = [out / f"{src.stem}_CH{t['ch']}.mp4" for t in tiles]
    parts = [d.with_suffix(".part.mp4") for d in dests]
    fc = f"[0:v]split={len(tiles)}" + "".join(f"[s{i}]" for i in range(len(tiles))) + ";" + ";".join(
        f"[s{i}]crop={t['w']}:{t['h']}:{t['x']}:{t['y']},fps={OUT_FPS}[o{i}]" for i, t in enumerate(tiles))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-filter_complex", fc]
    for i, (t, p) in enumerate(zip(tiles, parts)):
        cmd += ["-map", f"[o{i}]", "-an", *_encoder(t["w"], t["h"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(p)]
    cmd += ["-progress", "pipe:1", "-nostats"]
    log(f"{src.name}: {layout['grid'][0]}×{layout['grid'][1]} combo → "
        + ", ".join(f"CH{t['ch']} {t['w']}×{t['h']}" for t in tiles))
    dur = _duration(str(src))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last = -1
    for line in proc.stdout:
        if line.startswith("out_time_us=") and dur > 0:
            try:
                pct = min(99.9, int(line.split("=")[1]) / 1e6 / dur * 100)
            except ValueError:
                continue
            if pct - last >= 1:
                last = pct
                print(f"Progress: {pct:.1f}%", flush=True)
    err = proc.stderr.read()
    if proc.wait() != 0:
        for p in parts:
            p.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
    for p, d in zip(parts, dests):
        p.replace(d)
    print("Progress: 100%", flush=True)
    return dests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="folder for the per-pilot videos (default: next to the combo)")
    ap.add_argument("--channels", help="e.g. 1,3 (default: every channel with a pilot)")
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()
    layout = detect_layout(sample_frames(args.video))
    if args.detect_only:
        print(json.dumps(layout, indent=1))
        return
    try:
        for d in split_video(args.video, args.out, layout,
                             [int(c) for c in args.channels.split(",")] if args.channels else None):
            print(f"  {d}")
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
