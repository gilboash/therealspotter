"""
Gate IDs, laps and learning for race-day runs (fast: no video is read).

  label_run(tp, stem)        gate IDs + laps of one analysed run, with the
                             track's learned gates and timing
  learn(tp, stems, n_gates)  learn the track from analysed runs
                             (learn_track.py) and make it the track's memory
"""

import bisect
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import gate_decoder as gd
import learn_track as lt
from dataset_paths import TrackPaths


def _write_json(path: Path, data: dict):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def is_learned(tp: TrackPaths) -> bool:
    return tp.gate_memory.exists() and (tp.learned_timing.exists() or any(tp.gt_dir.glob("*.gt.json"))
                                        if tp.gt_dir.exists() else tp.learned_timing.exists())


def label_run(tp: TrackPaths, stem: str, g1_hint: Optional[List[float]] = None) -> dict:
    """Write gate IDs and laps into runs/<stem>.race_data.json (the per-frame
    overlays too) and return a short summary. The decoder's result is the
    system's (system_passes / system_laps); a review of the run, if there is
    one, is put on top (race_day/review.apply). g1_hint: the timing system's
    start/finish crossings in video seconds (gate_decoder.label_race)."""
    from race_day import review
    path = tp.race_data(stem)
    race = json.loads(path.read_text(encoding="utf-8"))
    source = race.get("candidate_passes") or race.get("system_passes") or race.get("passes", [])
    race["candidate_passes"] = source                    # the analysis output, kept for relabelling
    passes, laps, info = gd.label_race(source, tp.gate_memory, tp, g1_hint=g1_hint)
    race.update(passes=passes, laps=laps, system_passes=passes, system_laps=laps,
                gate_memory=str(tp.gate_memory), **info)
    review.apply(tp, stem, race)
    _overlays(race)
    _write_json(path, race)
    return {"passes": sum(p["gate_id"] >= 1 for p in race["passes"]), "laps": len(race["laps"])}


def apply_review(tp: TrackPaths, stem: str) -> Optional[dict]:
    """The run's review changed: its passes and laps become the reviewed ones
    (None: no analysis of this run)."""
    from race_day import review
    path = tp.race_data(stem)
    if not path.exists():
        return None
    race = json.loads(path.read_text(encoding="utf-8"))
    if review.apply(tp, stem, race):
        _overlays(race)
        _write_json(path, race)
    return {"passes": len(race.get("passes", [])), "laps": len(race.get("laps", [])), "review": race.get("review")}


def _overlays(race: dict):
    """Each frame's passes and laps (shown on the video), from race["passes"] / ["laps"]."""
    frames = race.get("frames", [])
    ts = [e["t"] for e in frames]
    for e in frames:
        e.pop("passes", None)
        e.pop("laps", None)
    if not frames:
        return

    def at(t):
        i = min(max(bisect.bisect_left(ts, t), 0), len(ts) - 1)
        if i > 0 and abs(ts[i - 1] - t) < abs(ts[i] - t):
            i -= 1
        return frames[i]
    for p in race.get("passes", []):
        at(p["t"]).setdefault("passes", []).append(p)
    for lp in race.get("laps", []):
        at(lp["t1"]).setdefault("laps", []).append({"lap": lp["lap"], "t": lp["t1"]})


def learn(tp: TrackPaths, stems: List[str], n_gates: int, verbose: bool = False,
          g1_times: Optional[Dict[str, List[float]]] = None) -> dict:
    """Learn the track's gates and timing from analysed runs and install it
    as the track's gate_memory.json. g1_times: per run, the timing system's
    start/finish crossings in video seconds (FPVTrackside's laps) — known G1
    moments, so learning needs only one full lap per run and is much faster."""
    videos, raw = [], {}
    for s in stems:
        race = json.loads(tp.race_data(s).read_text(encoding="utf-8"))
        passes = lt.learn_passes(race.get("candidate_passes") or race.get("system_passes") or race.get("passes", []))
        raw[s] = passes
        # Taken as they are: already on the video's clock. No align_lap_times here — sliding each run
        # onto whichever pass fits best lands on a neighbour of the start/finish (gates ~1 s apart).
        g1 = sorted(g1_times[s]) if g1_times and g1_times.get(s) else []
        videos.append(gd.VideoSeq(s, [gd.Obs(p["t"], gd._norm(p.get("query_embedding")),
                                              str(p.get("gate_type", "")), p.get("p_pass")) for p in passes],
                                  g1_times=g1))
    m, labels = lt.learn(videos, n_gates, lt.learn_cfg(), verbose=verbose)
    lt.save(tp, videos, raw, labels, m, n_gates)
    shutil.copy2(tp.learned_memory, tp.gate_memory)
    laps = [len(gd.build_laps([{"gate_id": g, "t": o.t} for o, g in zip(v.obs, labels[v.stem]) if g > 0], n_gates))
            for v in videos]
    return {"runs": len(stems), "laps": sum(laps),
            "legs_s": [round(float(gd.np.exp(m.log_mu[g])), 3) for g in range(1, n_gates + 1)]}
