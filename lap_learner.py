#!/usr/bin/env python3
"""
Lap-first track learning and labelling — for detectors that over-fire.

gate_decoder / learn_track assume the passes the detector finds are mostly the
real gates, so the first n+1 passes make a clean template lap. When the
detector fires a few extra passes every lap (a flagpole beside an arch, a gate
seen twice), that template is wrong and every lap boundary drifts.

Here the laps come first, and only the start gate closes a lap:

  1. laps      the first pass of a video is G1 (start/finish). Its later
               crossings are found by appearance (CLIP similarity to G1's
               images), one per lap, with the lap time as a prior once known.
               Nothing else can start or end a lap.
  2. points    the passes of every complete lap are aligned to each other —
               order, time since the previous matched pass, appearance — giving
               "pass points": places on the track where the detector fires.
  3. gates     G1 plus the n-1 most reliable points (seen in the most laps,
               then highest pass confidence) are the gates, in lap order. The
               other points are kept as tracked extras (E1, E2…): shown, never
               counted as gates.

Labelling a race video uses the same steps with the learned points: find the
laps from G1, then align each lap's passes to the points.

The learned model is saved in gate_timing.learned.json ("lap_model");
gate_decoder.label_race uses it when present, so extract_race / race_ui /
relabel_run pick it up without changes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

SIM_G1 = 0.88           # min CLIP similarity for a G1 crossing
G1_TIME_SD = 0.15       # lap-time prior when picking the next G1 (log-scale spread)
SIM_CENTER = 0.88       # appearance adds 0 at this similarity…
SIM_SCALE = 20.0        # …and this much per 1.0 of similarity above it
SIM_SCALE_REL = 30.0    # relative appearance (labelling): per 1.0 above the pass's mean similarity to all points
EXTRA_SKIP = -0.2       # leaving a tracked extra unmatched (they often don't fire)
LABEL_PASS_SKIP = -0.5  # labelling: a detected pass that fits no point (the detector over-fires; normal)
SCENE_SCALE = 80.0      # whole-frame look of a G1 crossing vs this video's other G1 crossings (per 1.0 of similarity)
LAP_SD = 0.3            # lap-time prior when labelling (log-scale; pilots differ in speed)…
LAP_SD_SLOW = 0.6       # …looser for a lap longer than usual (crash, slow launch) than for a short one
HEAD = 0.55             # the part of the lap (from G1) whose gate sequence identifies a G1 crossing
SPEEDS = (0.8, 0.87, 0.94, 1.0, 1.07, 1.15, 1.25)     # pilot speed vs the learned lap, tried per lap
G1_ACCEPT = 3.0         # min score for another G1 crossing (else the run ended)
EARLY_FRAC = 0.6        # take the earliest G1 candidate scoring at least this share of the best
DT_SD_ABS = 0.25        # spread (s) of an interval between two matched passes…
DT_SD_REL = 0.12        # …plus this fraction of the interval
GAP = -1.0              # leaving a pass or a point unmatched
MAX_STEP = 6            # max passes / points skipped between two matches
SLOW_CAP = 4.0          # max timing penalty for a gap longer than expected (a crash only costs this much)
MERGE_DT = 1.2          # points this close (s) that never share a lap are one point


@dataclass
class P:
    """One detected pass."""
    t: float
    emb: Optional[np.ndarray]
    p: float
    idx: int                                       # index in the video's sorted pass list
    gtype: str = ""
    scene: Optional[np.ndarray] = None             # CLIP of the whole frame just before the pass (scene_embed.py)


def from_passes(passes: List[dict]) -> List[P]:
    """Passes (race_data format, sorted by time) as P."""
    def nrm(e):
        if not e:
            return None
        v = np.asarray(e, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
    return [P(p["t"], nrm(p.get("query_embedding")), float(p.get("p_pass") or 1.0), i, str(p.get("gate_type", "")),
              nrm(p.get("scene_embedding"))) for i, p in enumerate(passes)]


@dataclass
class Point:
    offs: List[float] = field(default_factory=list)       # seconds after G1, at the median lap speed
    embs: List[np.ndarray] = field(default_factory=list)
    ps: List[float] = field(default_factory=list)
    laps: set = field(default_factory=set)
    members: list = field(default_factory=list)            # (video no, pass idx)
    types: list = field(default_factory=list)

    @property
    def off(self) -> float:
        return float(np.median(self.offs))

    def add(self, off: float, q: P, lap):
        self.offs.append(off)
        self.members.append((lap[0], q.idx))
        self.types.append(q.gtype)
        if q.emb is not None:
            self.embs.append(q.emb)
        self.ps.append(q.p)
        self.laps.add(lap)


def _sim(a: Optional[np.ndarray], bank) -> Optional[float]:
    if a is None or bank is None or len(bank) == 0:
        return None
    with np.errstate(all="ignore"):
        s = np.asarray(bank, dtype=np.float32) @ a
    return float(np.sort(s)[-3:].mean())


# ──────────────────────────────────────────────────────────────
# 1. Laps from the start gate
# ──────────────────────────────────────────────────────────────

def find_g1(ps: List[P], n: int, g1_bank=None, lap_s: Optional[float] = None, start: int = 0) -> List[int]:
    """Indices (into ps) of every G1 crossing, in time order, starting at
    `start`. Each next one is the pass most like G1 within one lap's reach;
    with a known lap time, crossings far from one lap later are penalised."""
    if not ps:
        return []
    i, chain = start, [start]
    bank = [ps[start].emb] if g1_bank is None else list(g1_bank)
    if g1_bank is None and ps[start].emb is None:
        return chain
    while True:
        cands = []
        for j in range(i + 1, len(ps)):
            dt = ps[j].t - ps[i].t
            if lap_s:
                if dt < 0.55 * lap_s:
                    continue
                if dt > 2.5 * lap_s:
                    break
            else:                                     # first lap: about n passes, allowing misses and extras
                if j - i < max(2, n - 3):
                    continue
                if j - i > 2 * n + 4:
                    break
            s = _sim(ps[j].emb, bank)
            if s is None or s < SIM_G1:
                continue
            score = (s - SIM_G1) * SIM_SCALE
            if lap_s:
                score -= (np.log(dt / lap_s) / G1_TIME_SD) ** 2
            cands.append((score, j))
        if not cands:
            return chain
        _, j = max(cands)
        chain.append(j)
        if g1_bank is None and ps[j].emb is not None:
            bank.append(ps[j].emb)
        i = j


def laps_of(ps: List[P], n: int, g1_bank=None, lap_s=None, start: int = 0) -> Tuple[List[int], float]:
    """G1 chain, refined once with this video's own median lap time."""
    chain = find_g1(ps, n, g1_bank, lap_s, start)
    if len(chain) >= 3:
        chain = find_g1(ps, n, g1_bank, float(np.median(np.diff([ps[c].t for c in chain]))), start)
    lap = float(np.median(np.diff([ps[c].t for c in chain]))) if len(chain) >= 2 else (lap_s or 0.0)
    return chain, lap


# ──────────────────────────────────────────────────────────────
# 2. Aligning a lap's passes to the pass points
# ──────────────────────────────────────────────────────────────

def align(offs: List[float], embs: List, pt_offs: List[float], pt_banks: List, end: Optional[float],
          relative: bool = False, pt_skip: Optional[List[float]] = None, with_score: bool = False,
          pass_skip: float = GAP):
    """Order-preserving alignment of a lap's passes (offsets in seconds after
    G1, at model speed) to the points (offsets, sorted). G1 is matched at 0 and,
    for a complete lap, the next G1 at `end`. A match scores its appearance and
    how well the time since the previous match fits the points' spacing.

    relative: appearance counts only as "more like this point than the others"
    (another camera shifts every similarity, so absolute values mean little).
    pt_skip: cost of leaving each point unmatched (default GAP); pass_skip: of leaving a pass unmatched.
    Returns the point index per pass, or -1 (and the score with with_score)."""
    A, B = len(offs), len(pt_offs)
    xo = [0.0] + list(offs) + ([end] if end is not None else [])     # anchors: G1 (and the next G1)
    po = [0.0] + list(pt_offs) + ([end] if end is not None else [])
    NA, NB = len(xo), len(po)
    app = np.zeros((NA, NB))
    for a in range(1, A + 1):
        sims = [_sim(embs[a - 1], pt_banks[b - 1]) for b in range(1, B + 1)]
        known = [x for x in sims if x is not None]
        for b, x in enumerate(sims, 1):
            if x is not None:
                app[a, b] = (x - np.mean(known)) * SIM_SCALE_REL if relative else (x - SIM_CENTER) * SIM_SCALE
    skip = np.zeros(NB + 1)                           # prefix sums of the point skip costs
    skip[2:B + 2] = np.cumsum(pt_skip if pt_skip is not None else [GAP] * B)
    if NB > B + 1:
        skip[NB] = skip[B + 1]
    NEG = -1e18
    best = np.full((NA, NB), NEG)
    back = {}
    best[0, 0] = 0.0
    last = (NA - 1, NB - 1) if end is not None else None
    for a in range(1, NA):
        for b in range(1, NB):
            if last and ((a == NA - 1) != (b == NB - 1)):
                continue                              # the closing G1 only pairs with itself
            for a0 in range(max(0, a - MAX_STEP - 1), a):
                for b0 in range(max(0, b - MAX_STEP - 1), b):
                    if best[a0, b0] <= NEG / 2:
                        continue
                    dt, dp = xo[a] - xo[a0], po[b] - po[b0]
                    sd = DT_SD_ABS + DT_SD_REL * max(dp, 0.0)
                    t_pen = 0.5 * ((dt - dp) / sd) ** 2
                    if dt > dp:                       # slower than expected (crash, recovery): capped
                        t_pen = min(t_pen, SLOW_CAP)
                    s = (best[a0, b0] + pass_skip * (a - a0 - 1) + (skip[b] - skip[b0 + 1])
                         - t_pen + app[a, b])
                    if s > best[a, b]:
                        best[a, b], back[(a, b)] = s, (a0, b0)
    if last:
        a, b = last
    else:                                             # open lap: best end, the rest unmatched
        cands = [(best[a, b] + pass_skip * (NA - 1 - a), a, b) for a in range(NA) for b in range(NB)
                 if best[a, b] > NEG / 2]
        _, a, b = max(cands)
    score = float(best[a, b]) + (pass_skip * (NA - 1 - a) if not last else 0.0)
    out = [-1] * A
    while (a, b) != (0, 0):
        if 1 <= a <= A and 1 <= b <= B:
            out[a - 1] = b - 1
        a, b = back[(a, b)]
    return (out, score) if with_score else out


# ──────────────────────────────────────────────────────────────
# 3. Learning: laps → points → gates
# ──────────────────────────────────────────────────────────────

def learn_points(videos: List[List[P]], n: int, g1_times: Optional[List[List[float]]] = None,
                 verbose: bool = True):
    """videos: each video's passes in time order. Returns
    (points, gate point indices in order, extra point indices, G1 chains, lap seconds)."""
    chains, laps = [], []                      # laps: (video no, lap no, passes, t0, t1)
    for vi, ps in enumerate(videos):
        if g1_times and g1_times[vi]:
            ts = np.array([q.t for q in ps])
            chain = sorted({int(np.argmin(np.abs(ts - g))) for g in g1_times[vi]})
        else:
            chain, _ = laps_of(ps, n)
        chains.append(chain)
        for k in range(len(chain) - 1):
            laps.append((vi, k, ps[chain[k] + 1: chain[k + 1]], ps[chain[k]].t, ps[chain[k + 1]].t))
        if verbose:
            ts = [ps[c].t for c in chain]
            print(f"  video {vi + 1}: G1 at " + ", ".join(f"{t:.1f}s" for t in ts)
                  + "  →  laps " + " / ".join(f"{b - a:.1f}s" for a, b in zip(ts, ts[1:])))
    if not laps:
        raise SystemExit("No complete lap found: the start gate (first pass) never came back.")
    lap_s = float(np.median([t1 - t0 for *_, t0, t1 in laps]))

    # start from the lap closest to the median length and pass count, then add the others one by one
    cnt = np.median([len(L[2]) for L in laps])
    ref = min(laps, key=lambda L: (abs(len(L[2]) - cnt), abs((L[4] - L[3]) - lap_s)))
    points: List[Point] = []
    for vi, k, lp, t0, t1 in [ref] + [L for L in laps if L is not ref]:
        sc = lap_s / (t1 - t0)
        offs = [(q.t - t0) * sc for q in lp]
        if points:
            banks = [np.array(p.embs) if p.embs else None for p in points]
            match = align(offs, [q.emb for q in lp], [p.off for p in points], banks, lap_s)
        else:
            match = [-1] * len(lp)
        # put the lap on the points' timeline: piecewise-linear between its matched passes
        # (a slow section then shifts only that section, not the rest of the lap)
        ax = [0.0] + [o for o, m in zip(offs, match) if m >= 0] + [lap_s]
        ay = [0.0] + [points[m].off for m in match if m >= 0] + [lap_s]
        for q, o, m in zip(lp, offs, match):
            if m >= 0:
                points[m].add(points[m].off, q, (vi, k))
            else:
                pt = Point()
                pt.add(float(np.interp(o, ax, ay)), q, (vi, k))
                points.append(pt)
        points.sort(key=lambda p: p.off)

    points = merge_points(points)
    nl = len(laps)
    ranked = sorted(range(len(points)), key=lambda i: (len(points[i].laps), float(np.mean(points[i].ps))),
                    reverse=True)
    gates = sorted(ranked[: n - 1], key=lambda i: points[i].off)
    extras = sorted(ranked[n - 1:], key=lambda i: points[i].off)
    g1 = Point()
    for vi, ps in enumerate(videos):
        for k, c in enumerate(chains[vi]):
            g1.add(0.0, ps[c], (vi, k))
    if verbose:
        print(f"  {nl} complete laps, median {lap_s:.1f}s; {len(points) + 1} pass points → "
              f"{min(n, len(points) + 1)} gates + {len(extras)} extras")
        print(f"    G1   at  0.0s  start/finish, {len(g1.laps)} crossings")
        for g, i in enumerate(gates, 2):
            p = points[i]
            print(f"    G{g:<3} at {p.off:4.1f}s  seen in {len(p.laps)}/{nl} laps  mean p {np.mean(p.ps):.2f}")
        for e, i in enumerate(extras, 1):
            p = points[i]
            print(f"    E{e:<3} at {p.off:4.1f}s  seen in {len(p.laps)}/{nl} laps  mean p {np.mean(p.ps):.2f}")
    return g1, [points[i] for i in gates], [points[i] for i in extras], chains, lap_s


def merge_points(points: List[Point], max_dt: float = MERGE_DT) -> List[Point]:
    """One point split over laps (a lap's pass landed a bit early or late, so it
    started a new point): merge neighbours that never share a lap, closest first."""
    points = sorted(points, key=lambda p: p.off)
    while True:
        best = None
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            d = b.off - a.off
            if d <= max_dt and not (a.laps & b.laps) and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            return points
        i = best[1]
        a, b = points[i], points[i + 1]
        a.offs += b.offs
        a.embs += b.embs
        a.ps += b.ps
        a.laps |= b.laps
        a.members += b.members
        a.types += b.types
        del points[i + 1]
        points.sort(key=lambda p: p.off)


def lap_model(g1: Point, gates: List[Point], extras: List[Point], lap_s: float, max_images: int = 24) -> dict:
    """The learned points as JSON: G1, the other gates in order, then the extras."""
    def one(p: Point, kind: str, no: int) -> dict:
        embs = p.embs
        if len(embs) > max_images:
            embs = [embs[i] for i in np.linspace(0, len(embs) - 1, max_images).round().astype(int)]
        return {"kind": kind, "id": no, "offset_s": round(p.off, 3), "seen_laps": len(p.laps),
                "p_mean": round(float(np.mean(p.ps)), 3), "embeds": [e.tolist() for e in embs]}
    pts = [one(g1, "gate", 1)] + [one(p, "gate", g) for g, p in enumerate(gates, 2)]
    pts += [one(p, "extra", e) for e, p in enumerate(extras, 1)]
    return {"version": 1, "lap_seconds": round(lap_s, 3), "points": pts}


# ──────────────────────────────────────────────────────────────
# Labelling a race video with a learned lap model
# ──────────────────────────────────────────────────────────────

def label(ps: List[P], model: dict, verbose: bool = False) -> Tuple[List[Tuple[str, int]], List[int], float]:
    """Per pass: ("gate", gate id) | ("extra", extra id) | ("none", 0).
    Also returns the G1 chain (indices into ps) and this video's lap time.

    Works across cameras (another camera shifts every CLIP similarity):
      - the first pass is G1 (start gate);
      - a later pass is a G1 crossing when the learned gate sequence starts right
        after it (G2, G3… at the learned spacing, at this pilot's speed), it looks
        more like G1 than like the other gates, and it comes about a lap later;
      - each lap's passes are aligned to the learned points by the time between
        consecutive matches, so a crash only disturbs one gap, not the whole lap."""
    out: List[Tuple[str, int]] = [("none", 0)] * len(ps)
    lap_s = float(model["lap_seconds"])
    if not ps:
        return out, [], lap_s
    rest = sorted(model["points"][1:], key=lambda p: p["offset_s"])       # gates and extras in lap order
    banks = [np.array(p["embeds"], dtype=np.float32) if p["embeds"] else None for p in rest]
    pt_offs = [p["offset_s"] for p in rest]
    pt_skip = [GAP if p["kind"] == "gate" else EXTRA_SKIP for p in rest]
    g1_bank = np.array(model["points"][0]["embeds"], dtype=np.float32) if model["points"][0]["embeds"] else None
    head = [k for k, o in enumerate(pt_offs) if o <= HEAD * lap_s]          # the first half of the lap

    def run(i: int, until: float, speed: float, points=None):
        """Open alignment of the passes after G1 at i, up to time `until`, at this speed."""
        idx = list(range(len(pt_offs))) if points is None else points
        lp = [k for k in range(i + 1, len(ps)) if ps[k].t < until]
        if not lp:
            return lp, [], 0.0
        m, score = align([(ps[k].t - ps[i].t) / speed for k in lp], [ps[k].emb for k in lp],
                         [pt_offs[b] for b in idx], [banks[b] for b in idx], None, relative=True,
                         pt_skip=[pt_skip[b] for b in idx], with_score=True, pass_skip=LABEL_PASS_SKIP)
        return lp, [idx[b] if b >= 0 else -1 for b in m], score

    def start_score(i: int) -> Tuple[float, float]:
        """How well the start of the learned lap follows pass i (best speed)."""
        best = (-1e9, 1.0)
        for sp in SPEEDS:
            _, _, sc = run(i, ps[i].t + HEAD * lap_s * sp, sp, head)
            best = max(best, (sc, sp))
        return best

    def g1_look(i: int) -> float:
        """More like G1 than like the most similar other point (camera-independent)."""
        s1 = _sim(ps[i].emb, g1_bank)
        others = [x for x in (_sim(ps[i].emb, b) for b in banks) if x is not None]
        return 0.0 if s1 is None or not others else (s1 - max(others)) * SIM_SCALE_REL

    scenes = np.array([q.scene if q.scene is not None else np.zeros(512, np.float32) for q in ps])
    has_scene = np.array([q.scene is not None for q in ps])

    def scene_look(j: int, ref: List[int]) -> float:
        """Does pass j look like the start gate's other crossings in THIS video (whole frame, same
        camera)? Relative to how alike a typical pass looks to them."""
        ref = [c for c in ref if c != j and has_scene[c]]
        if not ref or not has_scene[j]:
            return 0.0
        with np.errstate(all="ignore"):
            sims = scenes @ scenes[ref].T
        per = sims.mean(1)
        base = float(np.median(per[has_scene]))
        return float(per[j] - base) * SCENE_SCALE

    def search(fixed_ref: Optional[List[int]]):
        """The G1 chain. Scene reference: the crossings found so far, or a fixed set (second round)."""
        chain, speeds = [0], [start_score(0)[1]]
        own, i = lap_s * speeds[0], 0
        while True:
            ref = fixed_ref if fixed_ref is not None else chain
            cands = []
            for j in range(i + 1, len(ps)):
                dt = ps[j].t - ps[i].t
                if dt < 0.6 * own:
                    continue
                if dt > 2.2 * own:
                    break
                prior = (np.log(dt / own) / (LAP_SD_SLOW if dt > own else LAP_SD)) ** 2
                look = g1_look(j) + scene_look(j, ref)
                if ps[-1].t < ps[j].t + 0.25 * own:
                    # the video ends right after: no gate sequence to confirm it, so the looks and the lap time decide
                    total = G1_ACCEPT + look - 4 * prior if look > 0 else -1e9
                    sp = speeds[-1]
                else:
                    st, sp = start_score(j)
                    total = st + look - prior
                cands.append((total, j, sp))
            good = [c for c in cands if c[0] >= G1_ACCEPT]
            if not good:
                if verbose and cands:
                    b = max(cands)
                    print(f"  after G1 {ps[i].t:.1f}s: best next G1 {ps[b[1]].t:.1f}s scores only {b[0]:.1f} — run ended")
                break
            # the earliest strong one (a crossing a lap later beats the same gate two laps later); candidates
            # close together in time are one group, and the best of the group stands for it
            groups = []
            for c in sorted(good, key=lambda c: c[1]):
                if groups and ps[c[1]].t - ps[groups[-1][-1][1]].t <= 0.25 * own:
                    groups[-1].append(c)
                else:
                    groups.append([c])
            heads = [max(g) for g in groups]
            top = max(h[0] for h in heads)
            total, j, sp = next(h for h in heads if h[0] >= EARLY_FRAC * top)
            if verbose:
                print(f"  lap {len(chain)}: G1 {ps[i].t:.1f}s → {ps[j].t:.1f}s ({ps[j].t - ps[i].t:.1f}s)  score {total:.1f}")
            chain.append(j)
            speeds.append(sp)
            own = float(np.median(np.diff([ps[c].t for c in chain])))
            i = j
        return chain, speeds, own

    # round 1: the scene of each crossing against the ones found before it; round 2: against all of them
    # (the start gate's crossings must all look alike — the check done by eye on a contact sheet)
    chain, speeds, own = search(None)
    for _ in range(2):
        if not has_scene.any() or len(chain) < 2:
            break
        if verbose:
            print("  re-check against all the start-gate crossings found:")
        new = search(list(chain))
        if new[0] == chain:
            break
        chain, speeds, own = new
    for k, c in enumerate(chain):
        out[c] = ("gate", 1)
        until = ps[chain[k + 1]].t if k + 1 < len(chain) else ps[c].t + 1.2 * own
        lp, m, _ = run(c, until, speeds[k])
        for q, b in zip(lp, m):
            if b >= 0:
                out[q] = (rest[b]["kind"], int(rest[b]["id"]))
    return out, chain, own


# ──────────────────────────────────────────────────────────────
# Manual learning: the user marked the gates (web UI, Learn manual)
# ──────────────────────────────────────────────────────────────

def points_from_marks(videos: List[List[P]], marks: List[List[Tuple[float, int]]], tol: float = 0.6):
    """Gates from marked passes: marks[v] = [(time, gate id)…] with G1 = start/finish.
    Each mark takes the nearest unused detected pass within `tol` seconds (or
    stands alone, without an image, if the detector missed it). Laps run from
    one G1 mark to the next. Returns (g1, gates, extras=[], lap seconds, n)."""
    n = max((g for vm in marks for _, g in vm), default=0)
    if n < 2:
        raise ValueError("Mark the start gate (L) and at least one more gate (G).")
    pts = {g: Point() for g in range(1, n + 1)}
    lap_lens = []
    for vi, (ps, vm) in enumerate(zip(videos, marks)):
        vm = sorted(vm)
        used = set()
        snapped = []
        for t, g in vm:
            near = [(abs(q.t - t), k) for k, q in enumerate(ps) if abs(q.t - t) <= tol and k not in used]
            if near:
                k = min(near)[1]
                used.add(k)
                snapped.append((ps[k].t, g, ps[k]))
            else:
                snapped.append((t, g, P(t, None, 1.0, -1)))
        g1s = [t for t, g, _ in snapped if g == 1]
        lap_lens += list(np.diff(g1s))
        for t, g, q in snapped:
            k = max((i for i, a in enumerate(g1s) if a <= t), default=None)
            if k is None:
                continue                                  # before the first G1 mark
            lap = (vi, k)
            if g == 1:
                pts[1].add(0.0, q, lap)
            else:
                pts[g].add(t - g1s[k], q, lap)
    if not lap_lens:
        raise ValueError("Mark the start gate (L) at least twice — a full lap is needed.")
    lap_s = float(np.median(lap_lens))
    gates = [pts[g] for g in range(2, n + 1) if pts[g].offs]
    return pts[1], gates, [], lap_s, n


# ──────────────────────────────────────────────────────────────
# Learning from several videos (cameras differ): seed, then find the others' laps by the gate sequence
# ──────────────────────────────────────────────────────────────

SEED_CV = 0.12          # a video whose laps (by appearance alone) vary less than this is trusted as a seed


def learn_points_multi(videos: List[List[P]], n: int, g1_times: Optional[List[List[float]]] = None,
                       verbose: bool = True):
    """Like learn_points, but robust to videos from different cameras.

    Finding G1 by appearance alone works within one clean video, but on noisy
    feeds (analog, OSD, props in view) it can lock onto another gate. So:
      1. per video, laps by appearance; the videos whose laps are consistent
         (or that have known lap times) are the seeds;
      2. a first model is learned from the seeds;
      3. every other video's G1 crossings are found with that model (the gate
         sequence that follows G1 — label()), as in race mode;
      4. the final model is learned from all videos with those laps."""
    g1_times = g1_times or [[] for _ in videos]
    if len(videos) == 1:
        return learn_points(videos, n, g1_times, verbose)
    quality = []
    for vi, ps in enumerate(videos):
        if g1_times[vi]:
            quality.append(0.0)
            continue
        chain, _ = laps_of(ps, n)
        laps = np.diff([ps[c].t for c in chain])
        quality.append(float(np.std(laps) / np.mean(laps)) if len(laps) >= 2 else float("inf"))
    seeds = [vi for vi, q in enumerate(quality) if q <= SEED_CV]
    if not seeds:
        seeds = [int(np.argmin(quality))]
    if verbose:
        print("  lap consistency by appearance (CV): "
              + ", ".join(f"video {vi + 1} {q:.2f}" + (" seed" if vi in seeds else "") for vi, q in enumerate(quality)))
        print("  seed model:")
    g1, gates, extras, _, lap_s = learn_points([videos[vi] for vi in seeds], n,
                                               [g1_times[vi] for vi in seeds], verbose=False)
    model = lap_model(g1, gates, extras, lap_s)
    times = []
    for vi, ps in enumerate(videos):
        if vi in seeds or g1_times[vi]:
            times.append(g1_times[vi] or None)
            continue
        _, chain, _ = label(ps, model)
        times.append([ps[c].t for c in chain])
    # seeds keep their own appearance laps
    for vi in seeds:
        if not times[vi]:
            chain, _ = laps_of(videos[vi], n)
            times[vi] = [videos[vi][c].t for c in chain]
    if verbose:
        print("  laps found with the seed model, then learning from all videos:")
    return learn_points(videos, n, times, verbose)
