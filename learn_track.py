#!/usr/bin/env python3
"""
Learn a track from unreviewed videos — no teaching in learn_ui.

Given only the number of gates and that the first gate passed in every video
is G1 (start/finish), this learns everything gate_decoder.py needs to label
passes:

  timing      the duration of every gate-to-gate leg (G1→G2 … G13→G1)
  appearance  CLIP images of each gate (the gate memory)
  types       each gate's type (square / arch / …)

from the passes extract_race.py found in <track>/learn_videos/.

How: every video is labelled jointly with the same model, then the model is
re-estimated from those labels, and again until the labels stop changing
(an EM-style loop):

  1. start with no timing and no appearance: order + "first pass is G1" only,
     then let each video learn its own leg durations (laps repeat);
  2. pool all videos: each leg's duration (per-video speed factored out) and
     each gate's images from the passes labelled with it;
  3. relabel every video with the pooled timing + appearance; repeat.

Passes: a learn video's normal runs/<video>.race_data.json is used if it
exists (its gate labels are ignored). Otherwise learn_track runs the pass
analysis itself with a blank memory and saves runs/<video>.learn_passes.json
— never race_data.json, so Review and race_ui don't load a run without gates.

Outputs (track level):
  <track>/gate_memory.learned.json   gates in order with type and images
  <track>/gate_timing.learned.json   leg durations (log-normal per leg)

    python learn_track.py track1 --gates 13          # analyses new learn videos first (YOLO, ~1.5 min each)
    python learn_track.py track1 --gates 13 --evaluate   # test on the reviewed videos
    python learn_track.py track1 --gates 13 --install    # also make it the track's gate_memory.json
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import gate_decoder as gd
from dataset_paths import TrackPaths, list_tracks

MAX_IMAGES = 24         # varied images kept per gate (6 was too few to cover other pilots and light)


# ──────────────────────────────────────────────────────────────
# Loading the learn videos
# ──────────────────────────────────────────────────────────────

def read_lap_times(path: Path) -> List[float]:
    """G1 (start/finish) pass times in seconds from the video start, one per
    line (or comma/space separated). Lines that aren't numbers are ignored."""
    out = []
    for tok in path.read_text(encoding="utf-8").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return sorted(out)


def align_lap_times(g1: List[float], pass_times: List[float], tol: float = 0.35) -> Tuple[List[float], float]:
    """Lap timers count from the race start, not the video start: find the
    shift that puts the most G1 times on a detected pass (then refine it to the
    median offset of those matches)."""
    if not g1 or not pass_times:
        return g1, 0.0
    pt = np.array(pass_times)
    best = (-1, 0.0)
    for shift in np.unique(np.round(pt[:, None] - np.array(g1)[None, :], 2)):
        hits = sum(np.min(np.abs(pt - (a + shift))) <= tol for a in g1)
        if hits > best[0]:
            best = (hits, float(shift))
    shift = best[1]
    offs = [pt[np.argmin(np.abs(pt - (a + shift)))] - a for a in g1
            if np.min(np.abs(pt - (a + shift))) <= tol]
    shift = float(np.median(offs)) if offs else shift
    return [a + shift for a in g1], shift


def lap_file(video: Path) -> Optional[Path]:
    f = video.with_name(video.stem + ".laps.txt")
    return f if f.exists() else None


VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def learn_video_files(tp: TrackPaths) -> List[Path]:
    return sorted(p for p in tp.learn_videos.glob("*") if p.suffix.lower() in VIDEO_EXT)


def passes_file(tp: TrackPaths, stem: str) -> Optional[Path]:
    """A normal run if there is one (gate labels are ignored), else the learn run."""
    for p in (tp.race_data(stem), tp.learn_passes(stem)):
        if p.exists():
            return p
    return None


def analyse_learn_videos(tp: TrackPaths, n: int, det_model: str, clip_device: str):
    """Find the passes of every learn video that has none yet. A blank
    n-gate memory and greedy matching keep any taught track data out."""
    todo = [f for f in learn_video_files(tp) if passes_file(tp, f.stem) is None]
    if not todo:
        return
    import tempfile
    from extract_race import run_race_extraction
    blank = {"version": 2, "mode": "blank", "race_lookahead": 3, "expected_idx": 0, "max_embeds_per_gate": 6,
             "memory": [{"order_idx": i, "gate_id": i + 1, "gate_type": "unknown", "embeds": [],
                         "created_t": 0.0, "last_img": "", "embed_imgs": []} for i in range(n)]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(blank, tmp)
    try:
        for i, f in enumerate(todo, 1):
            print(f"  analysing {f.name} ({i}/{len(todo)})…")
            tp.runs_dir.mkdir(parents=True, exist_ok=True)
            run_race_extraction(str(f), det_model, tmp.name, str(tp.learn_passes(f.stem)),
                                clip_device=clip_device, gate_id_logic="greedy")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def load_learn_videos(tp: TrackPaths) -> Tuple[List[gd.VideoSeq], Dict[str, List[dict]]]:
    vids, raw = [], {}
    for f in learn_video_files(tp):
        rd = passes_file(tp, f.stem)
        if rd is None:
            print(f"  {f.name}: no passes found (skipped)")
            continue
        race = json.loads(rd.read_text(encoding="utf-8"))
        passes = sorted(race.get("passes", []), key=lambda p: p["t"])
        raw[f.stem] = passes
        lf = lap_file(f)
        g1 = []
        if lf:
            g1, shift = align_lap_times(read_lap_times(lf), [p["t"] for p in passes])
            print(f"  {f.name}: {len(g1)} G1 times from {lf.name}"
                  + (f" (shifted {shift:+.2f}s to the video)" if abs(shift) > 0.01 else ""))
        else:                                     # start-gate (L) marks from the web app's Manual mode
            mf = tp.runs_dir / f"{f.stem}.marks.json"
            if mf.exists():
                g1 = sorted(float(m["t"]) for m in json.loads(mf.read_text(encoding="utf-8")).get("marks", [])
                            if int(m.get("gate", 0)) == 1)
                if len(g1) >= 2:
                    print(f"  {f.name}: {len(g1)} G1 times from your start-gate marks")
                else:
                    g1 = []
        vids.append(gd.VideoSeq(f.stem, [gd.Obs(p["t"], gd._norm(p.get("query_embedding")),
                                                str(p.get("gate_type", "")), p.get("p_pass"))
                                         for p in passes], g1_times=g1))
    return vids, raw


# ──────────────────────────────────────────────────────────────
# Re-estimation from labels
# ──────────────────────────────────────────────────────────────

def _legs(v: gd.VideoSeq, labels: List[int], n: int):
    real = [(o.t, g) for o, g in zip(v.obs, labels) if g > 0]
    for (t0, g0), (t1, g1) in zip(real, real[1:]):
        if (g1 - g0) % n == 1 and t1 > t0:
            yield g0, float(np.log(t1 - t0))


def reestimate(videos: List[gd.VideoSeq], labels: Dict[str, List[int]], n: int, prev: gd.Model,
               use_appearance: bool) -> gd.Model:
    # per-video speed: median offset of its legs from the pooled leg medians
    per_leg = defaultdict(list)
    for v in videos:
        for g, x in _legs(v, labels[v.stem], n):
            per_leg[g].append(x)
    base = {g: float(np.median(x)) for g, x in per_leg.items()}
    speed = {}
    for v in videos:
        r = [x - base[g] for g, x in _legs(v, labels[v.stem], n) if g in base]
        speed[v.stem] = float(np.median(r)) if r else 0.0
    # legs with speed removed
    leg = defaultdict(list)
    for v in videos:
        for g, x in _legs(v, labels[v.stem], n):
            leg[g].append(x - speed[v.stem])
    log_mu, log_sd = prev.log_mu.copy(), prev.log_sd.copy()
    for g in range(1, n + 1):
        x = np.array(leg.get(g, []))
        if len(x) >= 3:
            med = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - med))       # robust: wrong labels don't widen it
            log_mu[g], log_sd[g] = float(med), float(max(0.2, mad))
    banks = [np.zeros((0, 512), dtype=np.float32) for _ in range(n)]
    if use_appearance:
        per_gate = defaultdict(list)
        for v in videos:
            for o, g in zip(v.obs, labels[v.stem]):
                if g > 0 and o.emb is not None:
                    per_gate[g].append(o.emb)
        for g, es in per_gate.items():
            banks[g - 1] = np.array(es, dtype=np.float32)
    m = gd.Model(n, prev.gate_types, banks, log_mu, log_sd, {})
    m.speed = speed
    return m


def _blank(n: int, log_mu=None, sd: float = 0.3) -> gd.Model:
    m = gd.Model(n, ["?"] * n, [np.zeros((0, 512), dtype=np.float32)] * n,
                 np.full(n + 1, np.log(1.5)) if log_mu is None else np.asarray(log_mu, dtype=float),
                 np.full(n + 1, sd), {})
    m.speed = {}
    return m


def first_lap_templates(videos: List[gd.VideoSeq], n: int) -> List[Tuple[str, np.ndarray]]:
    """Leg durations of each video's first lap, assuming it is clean: the first
    pass is G1 and the next n passes are G2…Gn, G1."""
    out = []
    for v in videos:
        if len(v.obs) > n:
            ts = [o.t for o in v.obs[: n + 1]]
            legs = np.diff(ts)
            if np.all(legs > 0.05):
                mu = np.zeros(n + 1)
                mu[1:] = np.log(legs)
                out.append((v.stem, mu))
    return out


def decode_all(videos, m: gd.Model, cfg: gd.Cfg, fixed_sd: Optional[float] = None):
    """Labels for every video plus the total log-score (with a fixed timing
    spread when comparing different templates fairly)."""
    mm = m if fixed_sd is None else gd.Model(m.n, m.gate_types, m.banks, m.log_mu, np.full(m.n + 1, fixed_sd), {})
    labels, total = {}, 0.0
    for v in videos:
        scale = float(np.exp(getattr(m, "speed", {}).get(v.stem, 0.0)))
        lab, sc = gd.decode(v, mm, cfg, scale, return_score=True)
        if not getattr(m, "speed", {}):           # first pass for this template: fit this video's speed
            for _ in range(2):
                scale = gd.estimate_scale(v, lab, mm)
                lab, sc = gd.decode(v, mm, cfg, scale, return_score=True)
        labels[v.stem], total = lab, total + sc
    return labels, total


def lap_template(videos: List[gd.VideoSeq], n: int) -> Optional[np.ndarray]:
    """With known G1 times: every lap holding exactly n passes is clean, so its
    passes are G1…Gn in order. Median leg durations over those laps (each lap
    scaled to the median lap time, so fast and slow pilots pool)."""
    laps = []
    for v in videos:
        g1 = v.g1_times
        for a, b in zip(g1, g1[1:]):
            ts = [o.t for o in v.obs if a - 0.35 <= o.t < b - 0.35]
            if len(ts) == n:
                laps.append(np.diff(ts + [b]))
    if len(laps) < 2:
        return None
    L = np.array(laps)
    L = L / L.sum(1, keepdims=True) * np.median(L.sum(1))
    mu = np.zeros(n + 1)
    mu[1:] = np.log(np.median(L, 0))
    return mu


def learn(videos: List[gd.VideoSeq], n: int, cfg: gd.Cfg, iters: int = 8, verbose: bool = True):
    """Returns (model, labels per video).

    1. Every video's first lap is a candidate template for the leg durations.
       Each candidate is refined on ALL videos with timing only (decode →
       re-estimate legs, speed per video), and the one that explains all
       videos best (fixed timing spread, so candidates compare fairly) wins.
    2. From that, timing and appearance are re-estimated together until the
       labels stop changing."""
    timing_cfg = gd.Cfg(**{**cfg.__dict__, "appearance": "none"})
    best = None
    with_laps = [v for v in videos if v.g1_times]
    mu = lap_template(with_laps, n) if with_laps else None
    candidates = [("clean laps between the known G1 times", mu)] if mu is not None else \
        first_lap_templates(videos, n)
    for stem, mu in candidates:
        m = _blank(n, mu)
        labels, _ = decode_all(videos, m, timing_cfg)
        for _ in range(4):
            m = reestimate(videos, labels, n, m, use_appearance=False)
            labels, _ = decode_all(videos, m, timing_cfg)
        _, score = decode_all(videos, m, timing_cfg, fixed_sd=0.3)
        if verbose:
            print(f"  template from {stem:12} → score {score:9.1f}")
        if best is None or score > best[0]:
            best = (score, stem, m, labels)
    if best is None:
        raise SystemExit(f"Need at least one learn video with a full lap ({n + 1}+ passes) to learn from.")
    _, stem, m, labels = best
    if verbose:
        print(f"  using the template from {stem}")
    for it in range(iters):
        m = reestimate(videos, labels, n, m, use_appearance=True)
        new, _ = decode_all(videos, m, cfg)
        changed = sum(sum(a != b for a, b in zip(new[s], labels[s])) for s in new)
        labels = new
        if verbose:
            print(f"  round {it + 1} (timing + appearance): {changed} labels changed")
        if changed == 0:
            break
    m = reestimate(videos, labels, n, m, use_appearance=True)
    return m, labels


# ──────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────

def _representatives(embs: np.ndarray, k: int) -> List[int]:
    """The k most typical images: highest mean similarity to the others,
    picked greedily so they aren't near-duplicates."""
    if len(embs) <= k:
        return list(range(len(embs)))
    with np.errstate(all="ignore"):
        S = embs @ embs.T
    central = S.mean(1)
    picked = [int(np.argmax(central))]
    while len(picked) < k:
        score = central - 0.5 * S[:, picked].max(1)
        score[picked] = -np.inf
        picked.append(int(np.argmax(score)))
    return picked


def save(tp: TrackPaths, videos, raw, labels, m: gd.Model, n: int):
    memory = []
    for g in range(1, n + 1):
        items = [(p, gd._norm(p.get("query_embedding"))) for v in videos
                 for p, lab in zip(raw[v.stem], labels[v.stem]) if lab == g and p.get("query_embedding")]
        types = Counter(str(p.get("gate_type", "")) for p, _ in items)
        if items:
            idx = _representatives(np.array([e for _, e in items], dtype=np.float32), MAX_IMAGES)
            chosen = [items[i] for i in idx]
        else:
            chosen = []
        memory.append({
            "order_idx": g - 1, "gate_id": g,
            "gate_type": types.most_common(1)[0][0] if types else "unknown",
            "embeds": [e.tolist() for _, e in chosen],
            "created_t": 0.0,
            "last_img": chosen[0][0].get("query_img", "") if chosen else "",
            "embed_imgs": [p.get("query_img", "") for p, _ in chosen],
            "learned_passes": len(items),
        })
    tp.learned_memory_own.write_text(json.dumps({
        "version": 2, "mode": "learned", "race_lookahead": 3, "expected_idx": 0,
        "max_embeds_per_gate": MAX_IMAGES, "learned_from": [v.stem for v in videos], "memory": memory,
    }, indent=2), encoding="utf-8")
    tp.learned_timing_own.write_text(json.dumps({
        "n_gates": n, "learned_from": [v.stem for v in videos],
        "log_mu": m.log_mu.tolist(), "log_sd": m.log_sd.tolist(),
        "leg_seconds": {f"G{g}->G{g % n + 1}": round(float(np.exp(m.log_mu[g])), 3) for g in range(1, n + 1)},
        "speed": {k: round(float(np.exp(v)), 3) for k, v in m.speed.items()},
    }, indent=2), encoding="utf-8")


def save_lap_model(tp: TrackPaths, stems: List[str], raw: Dict[str, List[dict]], g1, gates, extras,
                   lap_s: float, n: int, method: str = "manual") -> np.ndarray:
    """Write gate_memory.learned.json + gate_timing.learned.json from the gates
    the web app's Manual mode marked (lap_learner.points_from_marks): the same
    files as save(), plus the lap model the web app shows. Returns the leg durations."""
    import lap_learner as ll
    model = ll.lap_model(g1, gates, extras, lap_s, MAX_IMAGES)
    memory = []
    for g, pt in enumerate([g1] + gates, 1):
        items = [raw[stems[vi]][i] for vi, i in pt.members if i >= 0]
        items = [p for p in items if p.get("query_embedding")]
        if len(items) > MAX_IMAGES:
            items = [items[i] for i in _representatives(
                np.array([gd._norm(p["query_embedding"]) for p in items], dtype=np.float32), MAX_IMAGES)]
        types = Counter(t for t in pt.types if t)
        memory.append({
            "order_idx": g - 1, "gate_id": g,
            "gate_type": types.most_common(1)[0][0] if types else "unknown",
            "embeds": [gd._norm(p["query_embedding"]).tolist() for p in items],
            "created_t": 0.0,
            "last_img": items[0].get("query_img", "") if items else "",
            "embed_imgs": [p.get("query_img", "") for p in items],
            "learned_passes": len(pt.members), "offset_s": round(pt.off, 3),
        })
    tp.learned_memory_own.write_text(json.dumps({
        "version": 2, "mode": "learned", "race_lookahead": 3, "expected_idx": 0,
        "max_embeds_per_gate": MAX_IMAGES, "learned_from": stems, "memory": memory,
    }, indent=2), encoding="utf-8")
    # leg durations G1→G2 … Gn→G1 from the gate offsets, for gate_decoder's timing model
    offs = [0.0] + [p.off for p in gates] + [lap_s]
    legs = np.maximum(np.diff(offs), 0.05)
    log_mu, log_sd = np.zeros(n + 1), np.full(n + 1, 0.3)
    log_mu[1:len(legs) + 1] = np.log(legs)
    tp.learned_timing_own.write_text(json.dumps({
        "n_gates": n, "method": method, "learned_from": stems,
        "log_mu": log_mu.tolist(), "log_sd": log_sd.tolist(),
        "leg_seconds": {f"G{g}->G{g % n + 1}": round(float(legs[g - 1]), 3) for g in range(1, len(legs) + 1)},
        "lap_model": model,
    }), encoding="utf-8")
    return legs


# ──────────────────────────────────────────────────────────────
# Evaluation on reviewed videos (never used for learning)
# ──────────────────────────────────────────────────────────────

def evaluate(tp: TrackPaths, m: gd.Model, cfg: gd.Cfg, memory_banks=None):
    tests = gd.load_videos(tp)
    if memory_banks is not None:
        m = gd.Model(m.n, m.gate_types, memory_banks, m.log_mu, m.log_sd, {})
    C = T = 0
    rows = []
    for v in tests:
        if not any(o.label > 0 for o in v.obs):      # review not done yet
            continue
        labels = gd.decode_video(v, m, cfg)
        c = sum(1 for o, g in zip(v.obs, labels) if o.label > 0 and g == o.label)
        t = sum(1 for o in v.obs if o.label > 0)
        C, T = C + c, T + t
        rows.append((v.stem, c, t))
    return C, T, rows


def main():
    ap = argparse.ArgumentParser(description="Learn a track's gates and timing from unreviewed videos")
    ap.add_argument("track")
    ap.add_argument("--gates", type=int, required=True, help="Number of gates on the track")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--det-model", default=str(Path(__file__).parent / "cyn_current-20260715_best.pt"),
                    help="YOLO model for analysing new learn videos (use the one the pass scorer was trained on)")
    ap.add_argument("--clip-device", default=None, help="cpu / mps / cuda (default: auto)")
    ap.add_argument("--evaluate", action="store_true", help="Score the learned track on the reviewed videos")
    ap.add_argument("--install", action="store_true",
                    help="Make the learned memory the track's gate_memory.json (the old one is kept as .bak)")
    args = ap.parse_args()
    tp = next(t for t in list_tracks() if t.dir.name == args.track)
    if args.clip_device is None:
        import torch
        args.clip_device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    analyse_learn_videos(tp, args.gates, args.det_model, args.clip_device)
    videos, raw = load_learn_videos(tp)
    if not videos:
        ap.error(f"no analysed videos in {tp.learn_videos}")
    print(f"{tp.name}: learning {args.gates} gates from {len(videos)} videos "
          f"({sum(len(v.obs) for v in videos)} passes)")
    cfg = gd.Cfg(start_g1=10.0)
    m, labels = learn(videos, args.gates, cfg, args.iters)
    save(tp, videos, raw, labels, m, args.gates)
    for v in videos:
        seq = [g for g in labels[v.stem] if g > 0]
        laps = len(gd.build_laps([{"gate_id": g, "t": o.t} for o, g in zip(v.obs, labels[v.stem]) if g > 0],
                                 args.gates))
        print(f"  {v.stem:12} {len(seq)} passes labelled, {labels[v.stem].count(0)} judged false, "
              f"{laps} laps, speed ×{np.exp(m.speed.get(v.stem, 0.0)):.2f}")
    print("  legs (s): " + "  ".join(f"G{g}→{g % args.gates + 1} {np.exp(m.log_mu[g]):.2f}"
                                     for g in range(1, args.gates + 1)))
    print(f"Saved {tp.learned_memory.name} and {tp.learned_timing.name}")
    if args.evaluate:
        types, banks = gd.load_memory(tp.learned_memory)
        C, T, rows = evaluate(tp, m, gd.Cfg(), banks)
        print("\nOn the reviewed videos (not used for learning):")
        for s, c, t in rows:
            print(f"  {s:12} {c}/{t} = {100 * c / max(1, t):.0f}%")
        print(f"  gate ID {C}/{T} = {100 * C / max(1, T):.1f}%")
    if args.install:                     # the picked model's own gate memory (model_profiles.py)
        own = tp.gate_memory_own
        if own.exists():
            shutil.copy2(own, own.with_suffix(".json.bak"))
        shutil.copy2(tp.learned_memory_own, own)
        print(f"Installed → {own} (previous kept as {own.name}.bak)")


if __name__ == "__main__":
    main()
