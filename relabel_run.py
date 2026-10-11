"""
relabel_run.py — turn an existing analysis into race results without re-running YOLO.

Takes <track>/runs/<video>.learn_passes.json (or an existing race_data.json),
labels its passes with the track's current gate_memory.json + timing via
gate_decoder.label_race, and writes <track>/runs/<video>.race_data.json,
which race_ui loads when the video is opened.

    python relabel_run.py "<FPVdatasets>/<track>/learn_videos/<video>.mp4"
"""
import argparse
import bisect
import json
from pathlib import Path

import gate_decoder
from dataset_paths import track_of


def relabel(video: str) -> Path:
    tp = track_of(video)
    if tp is None:
        raise SystemExit(f"{video} is not inside a track folder")
    stem = Path(video).stem
    src = tp.race_data(stem) if tp.race_data(stem).exists() else tp.learn_passes(stem)
    if not src.exists():
        raise SystemExit(f"no analysis for {stem} in {tp.runs_dir}")
    import scene_embed
    if scene_embed.ensure_file(video, src):
        print(f"  added scene embeddings to {src.name}")
    data = json.loads(src.read_text(encoding="utf-8"))

    passes, laps, info = gate_decoder.label_race(data["passes"], tp.gate_memory, tp)
    data.update(info)

    # per-frame overlays (race_ui) follow the final passes and laps, as in extract_race
    frames = data["frames"]
    for e in frames:
        e.pop("passes", None)
        e.pop("laps", None)
    frame_ts = [e["t"] for e in frames]

    def frame_at(tt):
        i = min(max(bisect.bisect_left(frame_ts, tt), 0), len(frame_ts) - 1)
        if i > 0 and abs(frame_ts[i - 1] - tt) < abs(frame_ts[i] - tt):
            i -= 1
        return frames[i]
    for p in passes:
        frame_at(p["t"]).setdefault("passes", []).append(p)
    for lp in laps:
        frame_at(float(lp.get("t1", 0.0))).setdefault("laps", []).append(
            {"lap": int(lp.get("lap", 0)), "t": float(lp.get("t1", 0.0))})
    data.update({"passes": passes, "laps": laps})

    out = tp.race_data(stem)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Gate IDs ({info['gate_id_logic']}, timing: {info['timing']}): "
          f"{sum(p['gate_id'] >= 1 for p in passes)} gate passes, "
          f"{sum(p.get('source') == 'EXTRA' for p in passes)} extras, {len(laps)} laps → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Relabel an analysed video with the track's learned gates")
    ap.add_argument("video")
    relabel(ap.parse_args().video)
