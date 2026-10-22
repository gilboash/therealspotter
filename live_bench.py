#!/usr/bin/env python3
"""
live_bench.py — how small can the live spotter's detector get? (towards the Raspberry Pi 5)

For each detector input size: every frame of a few analysed videos through the live spotter (no frames dropped, so
only the size changes), passes and laps compared with the offline analysis; and the detector's speed per frame on
the GPU (MPS) and on 4 CPU threads (the Pi 5 has 4 cores and no GPU the detector can use).

    python live_bench.py                                   # default sizes and videos
    python live_bench.py --sizes 320 256 192 --videos 4
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import online_decoder
import online_spotter as osp
from dataset_paths import list_tracks

TRACKS = ("fril-27-first-track", "fril-26-race-7")


def videos(per_track: int):
    """The first analysed race videos of each track (skipping learn videos and runs of another track)."""
    out = []
    for tn in TRACKS:
        tp = next(t for t in list_tracks() if t.dir.name == tn)
        n = 0
        for f in sorted(tp.runs_dir.glob("*.race_data.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            stem = f.name[: -len(".race_data.json")]
            if stem.startswith("learn-") or "fril race 1 2027" in stem or len(d.get("laps", [])) < 2:
                continue
            v = next((p for p in tp.dir.rglob(stem + ".*") if p.suffix.lower() in (".mp4", ".mov", ".mkv")), None)
            if v is None:
                continue
            out.append((tp, v, d))
            n += 1
            if n >= per_track:
                break
    return out


def match(got, ref, window):
    used, m = set(), 0
    for r in sorted(ref):
        j = min((j for j in range(len(got)) if j not in used and abs(got[j] - r) <= window),
                key=lambda j: abs(got[j] - r), default=None)
        if j is not None:
            used.add(j)
            m += 1
    return m


def accuracy(size: int, vids, conf: float, det_model=osp.DET_MODEL):
    sp = osp.OnlineSpotter(det_model, size, conf, device=None if str(det_model).endswith(".pt") else "cpu")
    tot = {"ref": 0, "found": 0, "extra": 0, "laps_ref": 0, "laps_ok": 0, "gate_ok": 0, "gate_n": 0}
    for tp, v, d in vids:
        dec = online_decoder.for_track(tp)
        passes, laps, gates = [], [], []

        def on_event(e):
            if e["type"] == "pass":
                passes.append(e["t"])
            elif e["type"] == "lap":
                laps.append(e)
            elif e["type"] == "gate":
                gates.append(e)

        sp.run(osp.FrameSource(str(v), realtime=False).start(), on_event, decoder=dec)
        ref = [p["t"] for p in d["passes"] if p.get("source") != "NOPASS" and p.get("reason") != "near_miss"]
        m = match(passes, ref, 0.35)
        tot["ref"] += len(ref); tot["found"] += m; tot["extra"] += len(passes) - m
        rl = d.get("laps", [])
        tot["laps_ref"] += len(rl)
        used = set()
        for r in rl:
            k = min((k for k in range(len(laps)) if k not in used and abs(laps[k]["t1"] - r["t1"]) <= 0.3
                     and abs(laps[k]["dt"] - r["dt"]) <= 0.3), key=lambda k: abs(laps[k]["t1"] - r["t1"]), default=None)
            if k is not None:
                used.add(k)
        tot["laps_ok"] += len(used)
        named = [p for p in d["passes"] if p.get("source") == "RACE" and p.get("gate_id", -1) >= 1]
        for g in gates:
            r = min(named, key=lambda p: abs(p["t"] - g["t"]), default=None)
            if r is not None and abs(r["t"] - g["t"]) <= 0.35:
                tot["gate_n"] += 1
                tot["gate_ok"] += int(r["gate_id"] == g["gate_id"])
        print(f"    {size}: {v.name[:30]:30s} passes {m}/{len(ref)} (+{len(passes) - m})  laps {len(used)}/{len(rl)}",
              flush=True)
    return tot


def speed(size: int, video: Path, frames: int = 150):
    """Detector ms per frame on the GPU and on 4 CPU threads (a fresh model each: Ultralytics keeps its device)."""
    import torch
    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(video))
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    cap.release()
    out = {}
    for dev in ("mps", "cpu"):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        if dev == "cpu":
            torch.set_num_threads(4)
        det = YOLO(str(osp.DET_MODEL))
        for f in imgs[:10]:
            det(f, imgsz=size, verbose=False, device=dev)
        t0 = time.perf_counter()
        for f in imgs:
            det(f, imgsz=size, verbose=False, device=dev)
        out[dev] = 1000 * (time.perf_counter() - t0) / len(imgs)
    return out


def main():
    ap = argparse.ArgumentParser(description="Detector input size vs passes, laps and speed (live spotter)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 416, 320, 256, 192, 160])
    ap.add_argument("--videos", type=int, default=3, help="analysed race videos per track")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", help="save the results as JSON here")
    ap.add_argument("--det-model", default=str(osp.DET_MODEL),
                    help="detector: .pt, or an exported model (.onnx, *_ncnn_model) run on the CPU")
    ap.add_argument("--no-speed", action="store_true", help="accuracy only (time exported models with rt_bench)")
    args = ap.parse_args()
    vids = videos(args.videos)
    print(f"{len(vids)} videos: " + ", ".join(v.name for _, v, _ in vids))
    rows = []
    for size in args.sizes:
        acc = accuracy(size, vids, args.conf, args.det_model)
        sp = {} if args.no_speed else speed(size, vids[0][1])
        rows.append({"size": size, **acc, **{f"ms_{k}": round(v, 1) for k, v in sp.items()}})
        r = rows[-1]
        print(f"  {size}: passes {r['found']}/{r['ref']} (+{r['extra']} extra), gate names {r['gate_ok']}/{r['gate_n']}, "
              f"laps {r['laps_ok']}/{r['laps_ref']}; detector {r.get('ms_mps', '-')} ms GPU, {r.get('ms_cpu', '-')} ms CPU×4",
              flush=True)
    print("\n| Detector input | Passes found | Extra | Gate names right | Laps right (±0.3 s) | ms/frame GPU | ms/frame CPU ×4 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['size']} | {r['found']}/{r['ref']} ({100 * r['found'] / max(1, r['ref']):.0f}%) | {r['extra']} | "
              f"{100 * r['gate_ok'] / max(1, r['gate_n']):.1f}% | {r['laps_ok']}/{r['laps_ref']} | {r.get('ms_mps', '-')} | {r.get('ms_cpu', '-')} |")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
