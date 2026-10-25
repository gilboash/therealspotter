#!/usr/bin/env python3
"""
Gate identity by sequence decoding (PLAN.md Milestone 2).

The greedy GateDB match labels one pass at a time from appearance alone.
Here every pass of a video is labelled at once: the most likely sequence of
gates G1…GN given

  order       a pass is normally the next gate; skipping k gates, flying the
              same gate again, or a pass that isn't real cost extra
  timing      gate-to-gate durations learned from reviewed videos, scaled to
              this pilot's speed (estimated from the video itself)
  appearance  CLIP similarity of the pass crop to each gate's examples
  type        the gate type YOLO saw vs. each gate's type

A dynamic program finds the best labelling (observations may be marked as
false passes).  Every term is a log-probability, so cues add up.

With the pass scorer's two stages (pass_scorer.json has "decoder"), each
observation's chance of being false comes from its p_pass, the passes judged
false are dropped, and each pass's time is corrected by its gate's usual
offset ("time_offsets").

    python gate_decoder.py track1          # compare decoders, leave-one-video-out
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_paths import TrackPaths, list_tracks


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

@dataclass
class Obs:
    t: float
    emb: Optional[np.ndarray]
    gtype: str
    p_pass: Optional[float] = None
    label: int = -1                   # true gate id, 0 = not a real pass, -1 = unknown


@dataclass
class VideoSeq:
    stem: str
    obs: List[Obs]
    real: List[dict] = field(default_factory=list)   # reviewed real passes (t, gate_id)
    memory_video: bool = False
    g1_times: List[float] = field(default_factory=list)   # known G1 (lap) times, e.g. from a lap timer


def _norm(e) -> Optional[np.ndarray]:
    if e is None or len(e) == 0:
        return None
    v = np.asarray(e, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def load_memory(path: Path) -> Tuple[List[str], List[np.ndarray]]:
    mem = json.loads(path.read_text(encoding="utf-8")).get("memory", [])
    mem = sorted(mem, key=lambda g: int(g.get("order_idx", 0)))
    types = [str(g.get("gate_type", "?")) for g in mem]
    banks = [np.array([_norm(e) for e in g.get("embeds", []) if e], dtype=np.float32) for g in mem]
    return types, banks


def match_labels(obs: List[Obs], real: List[dict], tol: float = 0.5):
    pairs = sorted((abs(o.t - m["t"]), i, j) for i, o in enumerate(obs)
                   for j, m in enumerate(real) if abs(o.t - m["t"]) <= tol)
    used_o, used_r = set(), set()
    for o in obs:
        o.label = 0
    for _, i, j in pairs:
        if i not in used_o and j not in used_r:
            used_o.add(i)
            used_r.add(j)
            obs[i].label = int(real[j]["gate_id"])


def load_videos(tp: TrackPaths, source: str = "system_passes") -> List[VideoSeq]:
    vids = []
    mem_stems = {f.stem for f in tp.memory_videos.glob("*")} if tp.memory_videos.exists() else set()
    for g in sorted(tp.gt_dir.glob("*.gt.json")):
        stem = g.name[: -len(".gt.json")]
        if not tp.race_data(stem).exists():
            continue
        gt = json.loads(g.read_text(encoding="utf-8"))
        race = json.loads(tp.race_data(stem).read_text(encoding="utf-8"))
        passes = race.get(source) or race.get("system_passes") or race.get("passes", [])
        obs = [Obs(p["t"], _norm(p.get("query_embedding")), str(p.get("gate_type", "")), p.get("p_pass"))
               for p in sorted(passes, key=lambda p: p["t"])]
        real = sorted([m for m in gt.get("marks", []) if m["kind"] == "pass" and m.get("tag") != "unsure"
                       and m.get("verdict") != "pending"], key=lambda m: m["t"])
        v = VideoSeq(stem, obs, real, stem in mem_stems)
        match_labels(v.obs, v.real)
        vids.append(v)
    return vids


# ──────────────────────────────────────────────────────────────
# Learned pieces (fit on the training videos only)
# ──────────────────────────────────────────────────────────────

@dataclass
class Model:
    n: int
    gate_types: List[str]
    banks: List[np.ndarray]                       # appearance examples per gate
    log_mu: np.ndarray                            # per transition g→g+1: mean log duration
    log_sd: np.ndarray
    type_logp: Dict[Tuple[str, int], float]       # log P(observed type | gate)


def fit(train: List[VideoSeq], n: int, gate_types: List[str], mem_banks: List[np.ndarray],
        use_reviewed: bool) -> Model:
    durs = defaultdict(list)
    for v in train:
        for a, b in zip(v.real, v.real[1:]):
            if (b["gate_id"] - a["gate_id"]) % n == 1 and b["t"] > a["t"]:
                durs[a["gate_id"]].append(b["t"] - a["t"])
    log_mu, log_sd = np.zeros(n + 1), np.ones(n + 1)
    all_lsd = []
    for gid in range(1, n + 1):
        x = np.log(np.array(durs.get(gid) or [1.5]))
        log_mu[gid] = np.median(x)
        log_sd[gid] = max(0.2, float(np.std(x))) if len(x) > 2 else 0.5
        all_lsd.append(log_sd[gid])
    banks = [b.copy() for b in mem_banks]
    if use_reviewed:
        extra = defaultdict(list)
        for v in train:
            for o in v.obs:
                if o.label > 0 and o.emb is not None:
                    extra[o.label].append(o.emb)
        for gid, es in extra.items():
            b = np.array(es, dtype=np.float32)
            banks[gid - 1] = np.vstack([banks[gid - 1], b]) if len(banks[gid - 1]) else b
    counts = defaultdict(lambda: 1.0)       # Laplace smoothing
    tot = defaultdict(lambda: 4.0)
    for v in train:
        for o in v.obs:
            if o.label > 0:
                counts[(o.gtype, o.label)] += 1
                tot[o.label] += 1
    type_logp = {}
    for ty in {o.gtype for v in train for o in v.obs} | set(gate_types):
        for gid in range(1, n + 1):
            type_logp[(ty, gid)] = float(np.log(counts[(ty, gid)] / tot[gid]))
    return Model(n, gate_types, banks, log_mu, log_sd, type_logp)


# ──────────────────────────────────────────────────────────────
# Scoring terms
# ──────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    appearance: str = "memory"      # none | memory | reviewed
    temp: float = 0.02              # softmax temperature over gate similarities (lower = stronger)
    topk: int = 3                   # gate similarity = mean of its top-k example sims
    use_type: bool = False
    use_timing: bool = True
    order: bool = True              # False → each pass labelled on its own (argmax)
    p_skip: float = 0.02            # skipping k gates: p_skip^k
    p_repeat: float = 0.01          # same gate again
    p_false: float = 0.01           # an observation that isn't a real pass
    false_emit: float = 0.0         # log-score of a false pass's appearance (relative)
    start_g1: float = 3.0           # extra log-prior for the first real pass being G1
    speed_iters: int = 2            # re-estimate the pilot's speed this many times
    window: int = 4                 # max consecutive false observations
    g1_tol: float = 0.35            # a pass within this of a known G1 time is G1
    g1_weight: float = 6.0          # how strongly known G1 times pin the labels
    pass_weight: float = 0.0        # > 0: an observation with a p_pass is false with
    pass_bias: float = 0.0          #   logit(p) = pass_bias + pass_weight·logit(1 − p_pass)
    drop_false: bool = False        # label_race leaves out the observations judged false


def scorer_cfg() -> Cfg:
    """The decoder settings saved with the pass scorer (two stages), else the defaults."""
    import pass_scorer
    saved = pass_scorer.load_model() or {}
    dec = saved.get("decoder") or {}
    return Cfg(**{k: v for k, v in dec.items() if k in Cfg.__dataclass_fields__})


def emissions(v: VideoSeq, m: Model, cfg: Cfg) -> np.ndarray:
    """E[j, g] (g = 1…n): how much more likely the observation is under gate g
    than on average over gates (0 = no information), so observations with and
    without an image compete fairly against "false pass"."""
    E = np.zeros((len(v.obs), m.n + 1))
    for j, o in enumerate(v.obs):
        if cfg.appearance != "none" and o.emb is not None:
            with np.errstate(all="ignore"):     # macOS Accelerate matmul raises spurious FP warnings
                sims = np.array([np.sort(b @ o.emb)[-cfg.topk:].mean() if len(b) else 0.0 for b in m.banks])
            z = sims / cfg.temp
            z -= z.max()
            E[j, 1:] += z - np.log(np.exp(z).sum()) + np.log(m.n)
        if cfg.use_type:
            lp = np.array([m.type_logp.get((o.gtype, g), np.log(0.25)) for g in range(1, m.n + 1)])
            E[j, 1:] += lp - np.log(np.exp(lp).mean())
        if v.g1_times:              # known lap times: G1 exactly there, nowhere else
            near = min(abs(o.t - a) for a in v.g1_times) <= cfg.g1_tol
            E[j, 1] += cfg.g1_weight if near else -cfg.g1_weight
    return E


def _timing(m: Model, g0: int, d: int, dt: float, scale: float) -> float:
    """log-density of a duration dt for going from g0 forward d gates (d ≥ 1)."""
    if dt <= 0:
        return -20.0
    mu = sum(np.exp(m.log_mu[(g0 - 1 + k) % m.n + 1]) for k in range(d)) * scale
    sd = np.sqrt(sum(m.log_sd[(g0 - 1 + k) % m.n + 1] ** 2 for k in range(d)) / d)
    z = (np.log(dt) - np.log(mu)) / sd
    return float(-0.5 * z * z - np.log(sd))


def _false_logp(v: VideoSeq, cfg: Cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Per observation: log P(false), log P(real)."""
    lf = np.full(len(v.obs), np.log(cfg.p_false))
    for j, o in enumerate(v.obs):
        if cfg.pass_weight > 0 and o.p_pass is not None:
            p = min(max(float(o.p_pass), 1e-4), 1 - 1e-4)
            z = cfg.pass_bias + cfg.pass_weight * math.log((1 - p) / p)
            lf[j] = -math.log1p(math.exp(-z)) if z > -30 else z
    lr = np.log1p(-np.minimum(np.exp(lf), 1 - 1e-12))
    return lf, lr


def decode(v: VideoSeq, m: Model, cfg: Cfg, scale: float = 1.0, return_score: bool = False):
    """Labels (gate id, or 0 for a false pass) for every observation
    (and the best path's log-score with return_score)."""
    n, N = len(v.obs), m.n
    E = emissions(v, m, cfg)
    if not cfg.order:
        labels = [int(np.argmax(E[j, 1:]) + 1) for j in range(n)]
        return (labels, 0.0) if return_score else labels
    lf, lr = _false_logp(v, cfg)
    S = np.concatenate([[0.0], np.cumsum(lf)])           # S[k]: all of obs 0…k−1 false
    step = np.full(N, -np.inf)                       # log P(step d), d = 0…N-1
    skips = [cfg.p_skip ** (d - 1) for d in range(2, N)]   # skipping k gates: p_skip^k
    step[1] = np.log(max(1e-6, 1 - cfg.p_repeat - sum(skips)))
    step[0] = np.log(cfg.p_repeat)
    for d in range(2, N):
        step[d] = np.log(skips[d - 2])
    # every (previous gate g0, gate g) pair at once: steps and timing (as _timing)
    G0, G = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij")
    D = (G - G0) % N
    STEP = step[D]
    mu_g = np.exp(m.log_mu[1:N + 1])
    MU, SD = np.ones((N, N)), np.ones((N, N))
    for g0 in range(N):
        for d in range(1, N):
            idx = [(g0 + k) % N for k in range(d)]
            MU[g0, d] = mu_g[idx].sum() * scale
            SD[g0, d] = math.sqrt(sum(m.log_sd[i + 1] ** 2 for i in idx) / d)
    MUd, SDd, moved = MU[G0 - 1, D], SD[G0 - 1, D], D > 0
    NEG = -1e18
    best = np.full((n, N + 1), NEG)
    back = np.zeros((n, N + 1, 2), dtype=int)        # (previous obs index, previous gate); -1 = start
    for j in range(n):
        best[j, 1:] = S[j] + lr[j] + E[j, 1:]        # all earlier observations false
        best[j, 1] += cfg.start_g1
        back[j, 1:] = (-1, 0)
        for i in range(max(0, j - cfg.window - 1), j):
            alive = best[i, 1:] > NEG / 2
            if not alive.any():
                continue
            M = best[i, 1:, None] + (S[j] - S[i + 1]) + lr[j] + STEP + E[j, None, 1:]
            if cfg.use_timing:
                dt = v.obs[j].t - v.obs[i].t
                if dt <= 0:
                    T = np.full((N, N), -20.0)
                else:
                    z = (np.log(dt) - np.log(MUd)) / SDd
                    T = -0.5 * z * z - np.log(SDd)
                M = M + np.where(moved, T, 0.0)
            M[~alive] = NEG
            g0b = M.argmax(0)
            sb = M[g0b, np.arange(N)]
            upd = sb > best[j, 1:]
            best[j, 1:][upd] = sb[upd]
            back[j, 1:][upd] = np.stack([np.full(int(upd.sum()), i), g0b[upd] + 1], 1)
    # end: the remaining observations are false
    final = [(best[j, g] + S[n] - S[j + 1], j, g) for j in range(n) for g in range(1, N + 1)]
    if not final:
        return ([], 0.0) if return_score else []
    score, j, g = max(final)
    labels = [0] * n
    while j >= 0:
        labels[j] = g
        j, g = back[j, g]
    return (labels, float(score)) if return_score else labels


def estimate_scale(v: VideoSeq, labels: List[int], m: Model) -> float:
    r = []
    real = [(o.t, g) for o, g in zip(v.obs, labels) if g > 0]
    for (t0, g0), (t1, g1) in zip(real, real[1:]):
        if (g1 - g0) % m.n == 1 and t1 > t0:
            r.append(np.log(t1 - t0) - m.log_mu[g0])
    return float(np.exp(np.median(r))) if r else 1.0


def decode_video(v: VideoSeq, m: Model, cfg: Cfg) -> List[int]:
    scale = 1.0
    labels = decode(v, m, cfg, scale)
    if cfg.order and cfg.use_timing:
        for _ in range(cfg.speed_iters):
            scale = estimate_scale(v, labels, m)
            labels = decode(v, m, cfg, scale)
    return labels


def self_timing_labels(v: VideoSeq, m: Model, cfg: Cfg, iters: int = 4) -> List[int]:
    """No reviewed videos for this track: laps repeat, so learn the
    gate-to-gate durations from this video itself (decode → re-estimate)."""
    m.log_mu[:] = np.log(1.5)
    m.log_sd[:] = 1.0
    labels = decode(v, m, Cfg(**{**cfg.__dict__, "use_timing": False}))
    for _ in range(iters):
        d = defaultdict(list)
        real = [(o.t, g) for o, g in zip(v.obs, labels) if g > 0]
        for (t0, g0), (t1, g1) in zip(real, real[1:]):
            if (g1 - g0) % m.n == 1 and t1 > t0:
                d[g0].append(np.log(t1 - t0))
        for g, x in d.items():
            if len(x) >= 2:
                m.log_mu[g], m.log_sd[g] = float(np.median(x)), max(0.25, float(np.std(x)))
        labels = decode(v, m, cfg)
    return labels


# ──────────────────────────────────────────────────────────────
# Pipeline API (extract_race.py, score.py)
# ──────────────────────────────────────────────────────────────

def reviewed_sequences(tp: TrackPaths, exclude_stem: Optional[str] = None) -> List[VideoSeq]:
    """Reviewed gate sequences of a track (only the timing is needed for fit)."""
    out = []
    if tp is None or not tp.gt_dir.exists():
        return out
    for g in sorted(tp.gt_dir.glob("*.gt.json")):
        stem = g.name[: -len(".gt.json")]
        if stem == exclude_stem:
            continue
        gt = json.loads(g.read_text(encoding="utf-8"))
        real = sorted([m for m in gt.get("marks", []) if m["kind"] == "pass" and m.get("tag") != "unsure"
                       and m.get("verdict") != "pending"], key=lambda m: m["t"])
        if len(real) >= 3:
            out.append(VideoSeq(stem, [], real))
    return out


def _time_offsets(tp: Optional[TrackPaths]) -> Dict[str, float]:
    """Per-gate time offsets saved with the pass scorer for this track (by
    folder name), with "_all" for any other track; {} without a two-stage model."""
    import pass_scorer
    saved = pass_scorer.load_model() or {}
    offs = saved.get("time_offsets") or {}
    if not offs:
        return {}
    out = {"_all": float(offs.get("_all", 0.0))}
    if tp is not None:
        out.update({k: float(v) for k, v in (offs.get(tp.dir.name) or {}).items()})
    return out


def build_laps(passes: List[dict], n: int) -> List[dict]:
    """Laps from labelled passes, in GateDB's lap format. A lap starts at a G1
    pass; if G1 itself was missed, at the first pass after the order wraps.
    A new lap needs at least half the gates passed since the last start, so a
    G1 flown twice in a row doesn't count as a lap."""
    seq = [p for p in passes if p.get("gate_id", -1) >= 1]
    starts, prev = [], None
    for i, p in enumerate(seq):
        g = p["gate_id"]
        if (g == 1 or (prev is not None and g < prev)) and (not starts or i - starts[-1] >= n // 2):
            starts.append(i)
        elif g == 1 and starts and all(q["gate_id"] == 1 for q in seq[starts[-1]:i]):
            starts[-1] = i            # G1 flown again right away: the later crossing starts the lap
        prev = g
    laps = []
    for k, (a, b) in enumerate(zip(starts, starts[1:])):
        t0, t1 = seq[a]["t"], seq[b]["t"]
        splits, last = [], t0
        for idx, p in enumerate(seq[a:b]):
            splits.append({"idx": idx, "gate_id": p["gate_id"], "type": p.get("gate_type", ""), "t": p["t"],
                           "dt0": p["t"] - t0, "dprev": p["t"] - last, "recovered": bool(p.get("recovered"))})
            last = p["t"]
        laps.append({"lap": k + 1, "t0": t0, "t1": t1, "dt": t1 - t0, "splits": splits})
    return laps


def label_race(passes: List[dict], memory_path, tp: Optional[TrackPaths] = None,
               exclude_stem: Optional[str] = None, cfg: Optional[Cfg] = None,
               timing: str = "auto") -> Tuple[List[dict], List[dict], dict]:
    """Label every pass with its gate by sequence decoding.

    Timing comes from the track's reviewed videos (minus exclude_stem); with
    none, from the timing learn_track.py learned from learn_videos/; with
    neither, it is learned from this video. timing="reviewed" / "learned" /
    "self" forces one source. Returns (passes with gate_id / source
    / sim set, laps, info). Source is RACE for a labelled pass and NOPASS for
    one the decoder judges not real (left out with cfg.drop_false). Default
    cfg: the settings saved with the pass scorer. A pass's time is corrected
    by its gate's offset (saved with the pass scorer); "t_raw" keeps the
    original, so labelling again starts from it."""
    cfg = cfg or scorer_cfg()
    gate_types, banks = load_memory(Path(memory_path))
    n = len(gate_types)
    ordered = sorted(({**p, "t": p.get("t_raw", p["t"])} for p in passes), key=lambda p: p["t"])
    v = VideoSeq(exclude_stem or "", [Obs(p["t"], _norm(p.get("query_embedding")), str(p.get("gate_type", "")),
                                         p.get("p_pass")) for p in ordered])
    train = reviewed_sequences(tp, exclude_stem) if timing in ("auto", "reviewed") else []
    learned = tp.learned_timing if tp is not None and timing in ("auto", "learned") else None
    m = fit(train, n, gate_types, banks, use_reviewed=False)
    if train:
        labels, src = decode_video(v, m, cfg), f"reviewed videos ({len(train)})"
    elif learned is not None and learned.exists():
        d = json.loads(learned.read_text(encoding="utf-8"))
        if int(d.get("n_gates", n)) == n:
            m.log_mu, m.log_sd = np.array(d["log_mu"]), np.array(d["log_sd"])
            train = [None]                     # timing known: fit the pilot's speed below
            labels, src = decode_video(v, m, cfg), f"learned from {len(d.get('learned_from', []))} learn videos"
        else:
            labels, src = self_timing_labels(v, m, cfg), "learned from this video"
    else:
        labels, src = self_timing_labels(v, m, cfg), "learned from this video"
    out = []
    for p, o, g in zip(ordered, v.obs, labels):
        q = dict(p)
        q["gate_id"] = int(g) if g > 0 else -1
        q["source"] = "RACE" if g > 0 else "NOPASS"
        if g > 0 and o.emb is not None and len(banks[g - 1]):
            with np.errstate(all="ignore"):
                q["sim"] = round(float((banks[g - 1] @ o.emb).max()), 4)
        out.append(q)
    offsets = _time_offsets(tp)
    for q in out:
        off = offsets.get(str(q["gate_id"]), offsets.get("_all", 0.0)) if q["gate_id"] >= 1 else 0.0
        if off:
            q["t_raw"], q["t"] = q["t"], round(max(0.0, q["t"] + off), 4)
    if cfg.drop_false:
        out = [q for q in out if q["gate_id"] >= 1]
    out.sort(key=lambda q: q["t"])
    info = {"gate_id_logic": "sequence", "timing": src,
            "speed_scale": round(estimate_scale(v, labels, m), 3) if train else None}
    return out, build_laps(out, n), info


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(videos: List[VideoSeq], cfg: Cfg, gate_types, mem_banks, n: int, only_test: bool = True):
    """Leave-one-video-out. Returns (correct, real passes the detector found, per-video rows)."""
    rows, C, T, FP_ok, FP = [], 0, 0, 0, 0
    for v in videos:
        train = [o for o in videos if o.stem != v.stem]
        m = fit(train, n, gate_types, mem_banks, cfg.appearance == "reviewed")
        labels = decode_video(v, m, cfg)
        c = sum(1 for o, g in zip(v.obs, labels) if o.label > 0 and g == o.label)
        t = sum(1 for o in v.obs if o.label > 0)
        fpo = sum(1 for o, g in zip(v.obs, labels) if o.label == 0 and g == 0)
        fp = sum(1 for o in v.obs if o.label == 0)
        rows.append((v.stem, v.memory_video, c, t, fpo, fp))
        if not (only_test and v.memory_video):
            C, T, FP_ok, FP = C + c, T + t, FP_ok + fpo, FP + fp
    return C, T, FP_ok, FP, rows


def main():
    ap = argparse.ArgumentParser(description="Compare gate-identity decoders (leave-one-video-out)")
    ap.add_argument("track", nargs="?", default="track1")
    args = ap.parse_args()
    tp = next(t for t in list_tracks() if t.dir.name == args.track)
    gate_types, mem_banks = load_memory(tp.gate_memory)
    n = len(gate_types)
    videos = load_videos(tp)
    configs = {
        "appearance only (argmax)": Cfg(order=False),
        "order only": Cfg(appearance="none", use_timing=False),
        "order + timing": Cfg(appearance="none"),
        "order + appearance": Cfg(use_timing=False),
        "order + timing + appearance": Cfg(),
        "order + timing + reviewed appearance": Cfg(appearance="reviewed"),
    }
    print(f"{tp.name}: {len(videos)} videos, {n} gates (memory videos left out of the totals)\n")
    for name, cfg in configs.items():
        C, T, fpo, fp, _ = evaluate(videos, cfg, gate_types, mem_banks, n)
        print(f"  {name:<40} gate ID {C}/{T} = {100 * C / max(1, T):.1f}%   false passes spotted {fpo}/{fp}")


if __name__ == "__main__":
    main()
