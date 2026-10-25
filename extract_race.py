#!/usr/bin/env python3
"""
Headless race extraction for race_ui.py.

Runs the full pipeline (YOLO + tracker + PassDetector + CLIP + GateDB race matching)
on a video and saves per-frame data + race results to JSON.

Passes are decided post-race by pass_scorer.py (--pass-logic scorer, default):
while the video is read, the live PassDetector runs as before and every pass
candidate (a big track ending or suddenly shrinking) gets a CLIP embedding,
camera motion is measured (extract_motion.MotionEstimator).  After the last
frame the saved model (<data_root>/training/pass_scorer.json) picks the passes
with hindsight, and gate matching + laps are replayed over them in time order.
Gate IDs and laps then come from gate_decoder.py (--gate-id sequence, default):
all passes are labelled at once from gate order, gate-to-gate timing and
appearance.  "passes"/"laps" hold the result; "live_passes"/"live_laps" keep
the live detector's (with its greedy GateDB gate IDs) for comparison.  Without a saved model it falls back to the live
detector (--pass-logic live).

Usage (video inside a track folder — memory and output come from the track):
    python extract_race.py \
        --video  "<data_root>/track1/test_videos/myvideo.mp4" \
        --det-model  current_best_non_vocab.pt \
        --clip-device  mps
    → <data_root>/track1/runs/myvideo.race_data.json  (see dataset_paths.py)

--gate-memory / --output override the track defaults.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from dataset_paths import track_of
from extract_motion import MotionEstimator, save_motion
from pass_detector import PassDetector, detect_camera_edges
from gate_db import GateDB
from collections import deque
import pipeline_cfg
from lazy_spotter import (
    TimeTracker,
    ClipEmbedder,
    clamp_bbox,
    _get_yolo_names,
    _cls_to_name,
)


def _crop_padded(frame: np.ndarray, bbox: list, pad_frac: float = 0.5):
    """Crop frame to bbox expanded by pad_frac on each side. Returns (crop, padded_bbox)."""
    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(W, x2 + px), min(H, y2 + py)
    return frame[ny1:ny2, nx1:nx2], [nx1, ny1, nx2, ny2]


def _area_ratio(bbox, frame_area: float) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1) / max(frame_area, 1.0)


def run_race_extraction(
    video_path: str,
    det_model_path: str,
    gate_memory_path: str,
    output_json: str,
    det_conf: float = 0.25,
    clip_device: str = "cpu",
    pass_offset_sec: float = 0.0,
    sim_thresh: float = 0.88,
    min_match_margin: float = 0.03,
    g1_sim_thresh: Optional[float] = None,
    g1_margin: Optional[float] = None,
    require_same_type: bool = False,
    pass_logic: str = "scorer",
    gate_id_logic: str = "sequence",
    det_device: Optional[str] = None,
):
    print(f"Loading detector: {det_model_path}")
    det = YOLO(det_model_path)
    if det_device is None:              # a .pt on Apple Silicon: the GPU is ~3× faster with identical boxes
        import torch
        det_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)
    # a Core ML model (.mlpackage) picks its own hardware; a .pt runs on det_device
    # (Ultralytics' default is the CPU, even on Apple Silicon)
    det_kw = {} if str(det_model_path).endswith(".mlpackage") or not det_device else {"device": det_device}
    names = _get_yolo_names(det)
    print(f"Classes: {list(names.values())}")

    _cfg = pipeline_cfg.load()
    tracker = TimeTracker(**pipeline_cfg.tracker_kwargs(_cfg))
    passdet = PassDetector(**pipeline_cfg.gates_passdet_kwargs(_cfg))
    clip = ClipEmbedder(device=clip_device)

    gatedb = GateDB(
        sim_thresh=sim_thresh, require_same_type=require_same_type,
        min_lap_gap_sec=6.0, min_gates_between_laps=2,
        min_match_margin=min_match_margin, race_lookahead=3, max_embeds_per_gate=6,
        g1_sim_thresh=g1_sim_thresh, g1_margin=g1_margin,
    )
    gatedb.set_mode("race")
    gatedb.load_memory(gate_memory_path)
    print(f"[GateDB] Loaded {gatedb.memory_size()} gates from {gate_memory_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    ok, first_frame = cap.read()
    if ok:
        left_norm, right_norm = detect_camera_edges(first_frame)
        passdet.set_camera_edges(left_norm, right_norm)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / max(fps, 1.0)
    print(f"Video: {total_frames} frames  {fps:.1f} fps  {duration:.1f}s")
    print("Running race analysis…")

    frame_buffer: deque = deque(maxlen=5)
    bbox_buffer:  deque = deque(maxlen=5)   # {track_id: bbox} per frame, parallel to frame_buffer
    frames_data = []
    all_passes = []
    frame_idx = 0

    stem = Path(video_path).stem
    query_frames_dir = str(Path(output_json).parent / f"{stem}.race_query")
    Path(query_frames_dir).mkdir(parents=True, exist_ok=True)

    import pass_scorer
    saved_model = pass_scorer.load_model() if pass_logic == "scorer" else None
    two_stage = bool(saved_model and "decoder" in saved_model)   # v3: + "jump" candidates (pass_scorer)
    if pass_logic == "scorer" and saved_model is None:
        print(f"No pass scorer model at {pass_scorer.model_path()} — using the live pass detector")
        pass_logic = "live"
    if saved_model and Path(det_model_path).name not in saved_model.get("det_models", [Path(det_model_path).name]):
        print(f"WARNING: the pass scorer was trained on tracks from {', '.join(saved_model['det_models'])}, "
              f"not {Path(det_model_path).name} — its decisions may be worse than the live detector's")
    motion = MotionEstimator()
    cand_embeds = []                 # CLIP embedding for every pass candidate
    prev_tracks = {}                 # tid → (bbox, area_ratio, type) in the previous frame
    recent_area = {}                 # tid → deque of (t, area_ratio)
    prev_t = 0.0

    def embed_candidate(tid: int, t_c: float, gate_type: str):
        """Same crop as a live pass: the frame 4 back, padded box of that track."""
        frames, bboxes = list(frame_buffer), list(bbox_buffer)
        past = bboxes[0].get(tid) if bboxes else None
        if not frames or not past:
            return
        crop, _ = _crop_padded(frames[0], past)
        if crop.size == 0:
            return
        img = str(Path(query_frames_dir) / f"{stem}_c{frame_idx:06d}_{t_c:.3f}s.jpg")
        cv2.imwrite(img, crop)
        cand_embeds.append({"t": round(t_c, 4), "track_id": tid, "gate_type": gate_type,
                            "query_img": img, "query_embedding": clip.embed_bgr(crop).tolist()})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_idx += 1
        H, W = frame.shape[:2]
        frame_area = float(W * H)

        # ── YOLO ───────────────────────────────────────────────
        res = det(frame, conf=det_conf, verbose=False, max_det=50, **det_kw)[0]
        typed = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1, y1, x2, y2), W, H)
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = _cls_to_name(cls_id, names)
            gate_type = cls_name if conf >= 0.20 else "NONE"
            typed.append({"bbox": bb, "det_conf": conf, "type": gate_type, "type_score": conf})
        typed = sorted(typed, key=lambda d: d["det_conf"], reverse=True)[:10]

        # ── Track + pass detector ───────────────────────────────
        if pass_logic == "scorer":
            motion.update(frame)
        frame_buffer.append(frame)
        tracks = tracker.update(typed, t)
        passdet.update(tracks, t, frame_w=W, frame_h=H)
        bbox_buffer.append({int(tr.track_id): list(tr.bbox) for tr in tracks})

        st_map = getattr(passdet, "states", {}) or {}

        # ── Pass candidates for the scorer: big tracks that end or shrink ─
        if pass_logic == "scorer":
            cur = {int(tr.track_id): (list(tr.bbox), _area_ratio(tr.bbox, frame_area), str(tr.locked_type))
                   for tr in tracks}
            for tid, (bb, a, ty) in cur.items():
                dq = recent_area.setdefault(tid, deque())
                dq.append((t, a))
                while dq and dq[0][0] < t - 0.5:
                    dq.popleft()
                if tid in prev_tracks:
                    pb, pa = prev_tracks[tid][0], prev_tracks[tid][1]
                    if pa >= pass_scorer.BIG and (a <= pass_scorer.SHRINK * pa or
                                                  (two_stage and pass_scorer._iou(pb, bb) < pass_scorer.JUMP_IOU)):
                        embed_candidate(tid, prev_t, ty)
            for tid, (bb, pa, ty) in prev_tracks.items():
                if tid not in cur:
                    if max((x for _, x in recent_area.get(tid, [])), default=0.0) >= pass_scorer.BIG:
                        embed_candidate(tid, prev_t, ty)
                    recent_area.pop(tid, None)
            prev_tracks, prev_t = cur, t

        # ── Per-frame track info ────────────────────────────────
        frame_tracks = []
        for tr in tracks:
            st = st_map.get(int(tr.track_id))
            stage = str(getattr(st, "stage", "idle")) if st else "idle"
            last_area = float(getattr(st, "last_area", 0.0)) if st else 0.0
            last_cx = float(getattr(st, "last_cx", 0.0)) if st else 0.0
            last_cy = float(getattr(st, "last_cy", 0.0)) if st else 0.0
            area_ratio = last_area / max(frame_area, 1.0)
            cdist = (((last_cx / max(W, 1)) - 0.5) ** 2 + ((last_cy / max(H, 1)) - 0.5) ** 2) ** 0.5
            x1, y1, x2, y2 = tr.bbox
            frame_tracks.append({
                "track_id": int(tr.track_id),
                "bbox": [x1, y1, x2, y2],
                "type": str(tr.locked_type),
                "score": round(float(tr.score_ema), 3),
                "stage": stage,
                "area_ratio": round(area_ratio, 4),
                "cdist": round(float(cdist), 4),
            })

        # ── Pass events ─────────────────────────────────────────
        frame_passes = []
        frame_laps = []

        while True:
            evt = passdet.pop_any_passed()
            if evt is None:
                break

            tid = int(evt.get("track_id", -1))
            evt_type = str(evt.get("type", "UNKNOWN"))

            frames  = list(frame_buffer)
            bboxes  = list(bbox_buffer)
            embed_frame = frames[0] if frames else frame   # 4 frames before fire

            # Crop to gate bbox from that same frame; fall back to full frame
            past_bbox = bboxes[0].get(tid) if bboxes else None
            if past_bbox:
                embed_crop, _ = _crop_padded(embed_frame, past_bbox)
            else:
                embed_crop = embed_frame

            emb = clip.embed_bgr(embed_crop)

            q_fname = f"{stem}_q{frame_idx:06d}_{t:.3f}s.jpg"
            query_img_path = str(Path(query_frames_dir) / q_fname)
            cv2.imwrite(query_img_path, embed_crop)

            prev_race_laps = len(getattr(gatedb, "_race_laps", []))

            gid, sim, source, _s2, _mg, exp_before, _wsz = gatedb.race_match(
                now=t, gate_type=evt_type, emb=emb
            )
            if source != "RACE":
                gid = -1
            else:
                gatedb.on_pass(
                    now=t, gate_id=gid, gate_type=evt_type,
                    sim=sim, reason=str(evt.get("reason", "")),
                    track_id=tid,
                )

            new_race_laps = len(getattr(gatedb, "_race_laps", []))
            if new_race_laps > prev_race_laps:
                closed = gatedb._race_laps[-1]
                frame_laps.append({"lap": int(closed.get("lap", 0)), "t": float(closed.get("t1", t))})

            pass_entry = {
                "t": round(t, 4),
                "gate_id": int(gid),
                "gate_type": evt_type,
                "sim": round(float(sim), 4),
                "source": source,
                "reason": str(evt.get("reason", "")),
                "track_id": tid,
                "exp_before": int(exp_before),
                "query_img":  query_img_path,
                "query_embedding": emb.tolist(),
            }
            frame_passes.append(pass_entry)
            all_passes.append(dict(pass_entry))  # copy — offset shift must not affect frames_data

        # ── Save frame entry (skip empty frames to save space) ──
        entry = {"idx": frame_idx, "t": round(t, 4)}
        if frame_tracks:
            entry["tracks"] = frame_tracks
        if typed:                       # raw detections (web UI Detections window): x1 y1 x2 y2 type conf
            entry["dets"] = [[*d["bbox"], d["type"], round(d["det_conf"], 3)] for d in typed]
        if frame_passes:
            entry["passes"] = frame_passes
        if frame_laps:
            entry["laps"] = frame_laps
        frames_data.append(entry)

        if frame_idx % 150 == 0:
            pct = 100.0 * frame_idx / max(1, total_frames)
            print(f"Progress: {pct:.0f}%  frame={frame_idx}/{total_frames}  passes={len(all_passes)}")

    cap.release()

    # Shift pass timestamps back by offset so timeline ticks align with the
    # visual pass moment (PassDetector fires slightly after the actual pass)
    if pass_offset_sec > 0:
        for p in all_passes:
            p["t"] = round(max(0.0, p["t"] - pass_offset_sec), 4)
        print(f"Applied pass offset: -{pass_offset_sec:.2f}s to {len(all_passes)} pass events")

    race_laps = list(getattr(gatedb, "_race_laps", []))
    output = {
        "video": str(video_path),
        "gate_memory": str(gate_memory_path),
        "det_model": Path(det_model_path).name,
        "det_device": "coreml" if str(det_model_path).endswith(".mlpackage") else (det_device or "cpu"),
        "duration": float(duration),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "pass_logic": "live",
        "passes": all_passes,
        "laps": race_laps,
        "frames": frames_data,
    }

    final = all_passes
    if pass_logic == "scorer":
        motion_data = motion.result(video_path, fps)
        save_motion(motion_data, str(Path(output_json).parent / f"{stem}.motion.json"))
        decided = pass_scorer.decide_race({"frames": frames_data, "passes": all_passes},
                                          motion_data, W, H, saved_model, stem)
        if pass_offset_sec > 0:
            for p in decided:
                if p.get("source") == "NEW":
                    p["t"] = round(max(0.0, p["t"] - pass_offset_sec), 4)
        for p in decided:            # a new pass takes its candidate's embedding
            if p.get("query_embedding") is None:
                near = [c for c in cand_embeds if abs(c["t"] - p["t"]) <= 0.15]
                if near:            # prefer the same track, then the closest in time
                    c = min(near, key=lambda c: (c["track_id"] != p.get("track_id"), abs(c["t"] - p["t"])))
                    p["query_embedding"], p["query_img"] = c["query_embedding"], c["query_img"]
        from gate_db import replay_race
        final, _ = replay_race(decided, gate_memory_path, sim_thresh=sim_thresh,
                               min_match_margin=min_match_margin, g1_sim_thresh=g1_sim_thresh,
                               g1_margin=g1_margin, require_same_type=require_same_type)
        output.update({"pass_logic": f"scorer:{saved_model.get('logic')}", "candidates": len(cand_embeds)})
        n_new = sum(p.get("reason", "").startswith("scorer_") for p in final)
        print(f"Pass scorer: {len(final)} passes ({n_new} not fired by the live detector), "
              f"live detector had {len(all_passes)}")

    passes, laps = final, race_laps
    import scene_embed                   # whole-frame CLIP before each pass: the start gate's crossings look alike
    scene_embed.add_scene_embeddings(video_path, final, clip)
    if gate_id_logic == "sequence":
        import near_miss                 # + the pass scorer's near-misses, so a gate taken in a hard bank isn't lost
        passes, laps, info = near_miss.label(video_path, {"passes": final, "frames": frames_data,
                                                          "live_passes": all_passes},
                                             gate_memory_path, track_of(output_json))
        output.update(info)
        if two_stage:                    # the decoder drops the passes it judges false: relabelling starts from these
            output["candidate_passes"] = final
        print(f"Gate IDs by sequence decoding (timing: {info['timing']}): "
              f"{sum(p['gate_id'] >= 1 for p in passes)} passes labelled, {len(laps)} laps")
    else:
        output["gate_id_logic"] = "greedy"

    if passes is not all_passes:
        # per-frame overlays (race_ui) follow the final passes and laps
        frame_ts = [e["t"] for e in frames_data]
        for e in frames_data:
            if "passes" in e:
                e["live_passes"] = e.pop("passes")
            if "laps" in e:
                e["live_laps"] = e.pop("laps")
        import bisect

        def frame_at(tt):
            i = min(max(bisect.bisect_left(frame_ts, tt), 0), len(frame_ts) - 1)
            if i > 0 and abs(frame_ts[i - 1] - tt) < abs(frame_ts[i] - tt):
                i -= 1
            return frames_data[i]
        for p in passes:
            frame_at(p["t"]).setdefault("passes", []).append(p)
        for lp in laps:
            frame_at(float(lp.get("t1", 0.0))).setdefault("laps", []).append(
                {"lap": int(lp.get("lap", 0)), "t": float(lp.get("t1", 0.0))})
        output.update({"passes": passes, "laps": laps, "live_passes": all_passes, "live_laps": race_laps})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(output['passes'])} passes  {len(output['laps'])} laps ({output['pass_logic']}) → {output_json}")
    return len(all_passes)


def main():
    parser = argparse.ArgumentParser(description="Headless race extraction for race_ui.py")
    parser.add_argument("--video",             required=True,              help="Path to video file")
    parser.add_argument("--det-model",         required=True,              help="Path to YOLO .pt model")
    parser.add_argument("--gate-memory",       default=None,  help="gate_memory.json (default: the video's track)")
    parser.add_argument("--output",            default=None,  help="Output JSON (default: <track>/runs/<video>.race_data.json)")
    parser.add_argument("--det-conf",        type=float, default=0.25)
    parser.add_argument("--clip-device",     default="cpu",            help="cpu / mps / cuda")
    parser.add_argument("--det-device",      default=None,  help="cpu / mps / cuda for a .pt detector (default: mps or cuda when available, else cpu)")
    parser.add_argument("--pass-offset-sec", type=float, default=0.0,
                        help="Shift pass event timestamps back by this many seconds to align "
                             "timeline ticks with the visual pass moment (default: 0.0)")
    parser.add_argument("--sim-thresh",        type=float, default=0.88,
                        help="Minimum cosine similarity for a gate to count as matched (default: 0.88)")
    parser.add_argument("--min-match-margin",  type=float, default=0.03,
                        help="Minimum gap between best and second-best gate similarity (default: 0.03)")
    parser.add_argument("--g1-sim-thresh",     type=float, default=None,
                        help="Minimum cosine similarity for G1 (start gate); defaults to --sim-thresh")
    parser.add_argument("--g1-margin",         type=float, default=None,
                        help="Minimum margin for G1 (start gate); defaults to --min-match-margin")
    parser.add_argument("--pass-logic", choices=["scorer", "live"], default="scorer",
                        help="scorer: passes decided post-race by pass_scorer.py (saved model); "
                             "live: the live PassDetector's passes")
    parser.add_argument("--gate-id", choices=["sequence", "greedy"], default="sequence",
                        help="sequence: label all passes at once from gate order + timing + appearance "
                             "(gate_decoder.py); greedy: GateDB one pass at a time")
    parser.add_argument("--require-same-type", action="store_true", default=False,
                        help="Only match a detected gate against memory slots of the same type")
    args = parser.parse_args()

    tp = track_of(args.video)
    if tp is None and (args.gate_memory is None or args.output is None):
        parser.error("video is not inside a track folder — pass --gate-memory and --output")
    if args.gate_memory is None:
        args.gate_memory = str(tp.gate_memory)
    if args.output is None:
        tp.runs_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(tp.race_data(Path(args.video).stem))

    run_race_extraction(
        video_path=args.video,
        det_model_path=args.det_model,
        gate_memory_path=args.gate_memory,
        output_json=args.output,
        det_conf=args.det_conf,
        clip_device=args.clip_device,
        pass_offset_sec=args.pass_offset_sec,
        sim_thresh=args.sim_thresh,
        min_match_margin=args.min_match_margin,
        g1_sim_thresh=args.g1_sim_thresh,
        g1_margin=args.g1_margin,
        require_same_type=args.require_same_type,
        pass_logic=args.pass_logic,
        gate_id_logic=args.gate_id,
        det_device=args.det_device,
    )


if __name__ == "__main__":
    main()
