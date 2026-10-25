#!/usr/bin/env python3
"""
Pass detector v2 — decide each pass after the fact (post-race).

The live PassDetector decides at one instant.  Here every pass is decided
with hindsight from its candidate's approach, exit and what happens after:

  candidates   every pass the live detector fired, every track that ends
               while big (area ≥ 0.08 over its last 0.5 s), and every
               one-frame shrink of ≥35% while big; merged within 0.1 s
  features     approach of the GATE, stitched across track IDs (duration,
               alignment, growth, centring, angle of attack from the box's
               width/height), exit (borders touched,
               final size), after (does the same gate continue, next gate),
               camera motion from extract_motion.py (forward motion, turn,
               chaos)
  logics       several decision logics, compared on the reviewed videos with
               leave-one-video-out (LOGICS below)

Input per video: runs/<video>.race_data.json and runs/<video>.motion.json
(computed if missing).  No YOLO.

    python pass_scorer.py track1                 # compare all logics
    python pass_scorer.py track1 --logic tracks_motion --list
    python pass_scorer.py track1 --save          # train on all videos → training/pass_scorer.json
"""

import argparse
import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dataset_paths import TrackPaths, list_tracks, training_dir

TOL = 0.15          # on-time window, same as score.py
MERGE = 0.10        # candidates closer than this are one event
MERGE_PICK = "biggest"   # event time: "latest" exit among its big pieces, or the "biggest" piece
NMS = 0.25          # at most one pass per this window
BIG = 0.08          # area_ratio a track must reach to be a candidate
SHRINK = 0.65       # one-frame area drop to this fraction counts as an exit
TYPES = ("square", "arch", "circle", "flagpole")
STITCH = True       # judge the approach per gate: link overlapping boxes across track IDs
BEST_LOGIC = "tracks_motion"       # best on track1 leave-one-video-out, 11 videos (see README)
# Two stages (a saved model with "decoder" — v3, cozYnut's, model_profiles.py): the scorer keeps every candidate
# above a low threshold and gate_decoder drops the ones that don't fit the track. Such a model also gets "jump"
# candidates (the tracker slid on to the gate right behind), one pass per NMS_TWO_STAGE (gates can be 0.2 s
# apart) and the motion noise over moving frames only. A one-threshold model (v2) runs exactly as before.
NMS_TWO_STAGE = 0.15
JUMP_IOU = 0.7      # a big box overlapping its previous box less than this: the tracker moved on — an exit
STILL_RESID = 1e-4  # motion residual of a frame with no picture (gray feed) or a pilot standing still
TYPICAL_RESID = 0.02


# ──────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────

@dataclass
class Video:
    stem: str
    tp: Optional[TrackPaths]
    race: dict
    motion: dict
    W: int
    H: int
    cam: Tuple[float, float]
    tracks: Dict[int, list] = field(default_factory=dict)   # tid → [(t, bbox, area, cdist, stage, type, score)]
    by_frame: List[list] = field(default_factory=list)       # per saved frame: [(tid, t, bbox, area, cdist, stage, type, score)]
    frame_ts: List[float] = field(default_factory=list)
    gt: Optional[dict] = None
    two_stage: bool = False          # decided by a two-stage model (see NMS_TWO_STAGE)


def _video_file(tp: TrackPaths, stem: str) -> Optional[Path]:
    for d in (tp.test_videos, tp.memory_videos, tp.learn_videos):
        for f in d.glob(stem + ".*"):
            if f.stem == stem:
                return f
    return None


def load_video(tp: TrackPaths, stem: str, with_gt: bool = True) -> Video:
    race = json.loads(tp.race_data(stem).read_text(encoding="utf-8"))
    vf = _video_file(tp, stem)
    mpath = tp.motion(stem)
    if not mpath.exists():
        if vf is None:
            raise FileNotFoundError(f"no motion file and no video for {stem}")
        from extract_motion import extract_motion
        extract_motion(str(vf), str(mpath))
    motion = json.loads(mpath.read_text(encoding="utf-8"))
    W, H = 960, 540
    if vf is not None:
        cap = cv2.VideoCapture(str(vf))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or W
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or H
        cap.release()
    v = build_video(stem, race, motion, W, H, tp)
    gp = tp.gt(stem)
    if with_gt and gp.exists():
        v.gt = json.loads(gp.read_text(encoding="utf-8"))
    return v


def build_video(stem: str, race: dict, motion: dict, W: int, H: int, tp: Optional[TrackPaths] = None) -> Video:
    """A Video from in-memory race_data + motion dicts (no files needed)."""
    v = Video(stem, tp, race, motion, W, H, tuple(motion.get("camera_x", (0.0, 1.0))))
    for f in race.get("frames", []):
        v.frame_ts.append(f["t"])
        row = []
        for x in f.get("tracks", []):
            e = (f["t"], x["bbox"], x["area_ratio"], x["cdist"], x["stage"], x["type"], x["score"])
            v.tracks.setdefault(x["track_id"], []).append(e)
            row.append((x["track_id"],) + e)
        v.by_frame.append(row)
    return v


def live_passes(race: dict) -> List[dict]:
    """The live PassDetector's passes. Newer race_data keeps them in
    "live_passes" (its "passes" are this scorer's decisions)."""
    return race.get("live_passes", race.get("passes", []))


def real_passes(v: Video) -> List[dict]:
    """Reviewed real passes; unsure and still-pending marks are left out."""
    return [m for m in (v.gt or {}).get("marks", [])
            if m["kind"] == "pass" and m.get("tag") != "unsure" and m.get("verdict") != "pending"]


def ignored_times(v: Video) -> List[float]:
    g = v.gt or {}
    return ([m["t"] for m in g.get("marks", []) if m.get("tag") == "unsure" or m.get("verdict") == "pending"]
            + [fp["t"] for fp in g.get("false_passes", []) if fp.get("tag") == "unsure"])


# ──────────────────────────────────────────────────────────────
# Candidates
# ──────────────────────────────────────────────────────────────

@dataclass
class Cand:
    t: float
    tid: int
    det: Optional[dict] = None       # the live detector's pass, if it fired here
    kind: str = "end"                # end | shrink | jump (two stages) | det
    gtype: str = ""
    feats: Dict[str, float] = field(default_factory=dict)
    p: float = 0.0


def candidates(v: Video) -> List[Cand]:
    out: List[Cand] = []
    jumps: List[Cand] = []
    for tid, s in v.tracks.items():
        tE = s[-1][0]
        if max(a for t, _, a, *_ in s if t >= tE - 0.5) >= BIG:
            out.append(Cand(tE, tid, kind="end"))
        for x0, x1 in zip(s, s[1:]):
            if x0[2] >= BIG and x1[2] <= SHRINK * x0[2]:
                out.append(Cand(x0[0], tid, kind="shrink"))
            elif v.two_stage and x0[2] >= BIG and _iou(x0[1], x1[1]) < JUMP_IOU and x1[0] - x0[0] < 0.1:
                jumps.append(Cand(x0[0], tid, kind="jump"))
    for p in live_passes(v.race):
        out.append(Cand(p["t"], int(p.get("track_id", -1)), det=p, kind="det"))
    # merge within MERGE: pieces of one gate end at different moments — the pass is
    # when the last big piece leaves; remember any detector pass
    out.sort(key=lambda c: c.t)
    merged: List[List[Cand]] = []
    for c in out:
        if merged and c.t - merged[-1][-1].t <= MERGE:
            merged[-1].append(c)
        else:
            merged.append([c])
    res = []
    for grp in merged:
        own = [c for c in grp if c.kind != "det"]
        det = next((c.det for c in grp if c.det is not None), None)
        if own:
            amax = max(_area_near(v, c.tid, c.t) for c in own)
            if MERGE_PICK == "latest":
                best = max((c for c in own if _area_near(v, c.tid, c.t) >= 0.5 * amax), key=lambda c: c.t)
            else:
                best = max(own, key=lambda c: _area_near(v, c.tid, c.t))
        else:
            best = grp[0]
        res.append(Cand(best.t, best.tid, det=det, kind=best.kind if own else "det"))
    if not jumps:
        return res
    # a jump right where another candidate already is adds nothing
    res += [c for c in jumps if not any(abs(o.t - c.t) <= MERGE for o in res)]
    return sorted(res, key=lambda c: c.t)


def _area_near(v: Video, tid: int, t: float) -> float:
    s = v.tracks.get(tid)
    if not s:
        return 0.0
    return max((a for tt, _, a, *_ in s if t - 0.3 <= tt <= t + 1e-6), default=0.0)


# ──────────────────────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _borders(v: Video, b, tol: float = 0.03) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    L, R = v.cam[0] * v.W, v.cam[1] * v.W
    return (int(x1 <= L + tol * v.W), int(x2 >= R - tol * v.W),
            int(y1 <= tol * v.H), int(y2 >= v.H - tol * v.H))


def type_ratios(videos: List[Video]) -> Dict[str, float]:
    """Front-on width/height per gate type: the upper end of boxes not touching the border."""
    rs: Dict[str, List[float]] = {}
    for v in videos:
        for s in v.tracks.values():
            for _, b, a, c, _, ty, _ in s:
                if a >= 0.02 and not any(_borders(v, b)):
                    rs.setdefault(ty, []).append((b[2] - b[0]) / max(1.0, b[3] - b[1]))
    return {ty: float(np.percentile(r, 90)) for ty, r in rs.items() if len(r) >= 20}


def _mwin(v: Video, key: str, t0: float, t1: float) -> np.ndarray:
    arr = v.motion[key]
    fps = v.motion.get("fps", 30.0)
    i0, i1 = max(0, int(round(t0 * fps))), min(len(arr), int(round(t1 * fps)) + 1)
    return np.asarray(arr[i0:i1], dtype=float) if i1 > i0 else np.zeros(1)


def stitched(v: Video, tid: int, t_end: float, back: float = 1.0) -> list:
    """The gate's history before t_end across track IDs: walk back frame by
    frame and link the box of the same type that overlaps the chain's box."""
    s = [x for x in v.tracks.get(tid, []) if x[0] <= t_end + 1e-6]
    if not s:
        return []
    chain = [s[-1]]
    ty = s[-1][5]
    i = bisect.bisect_left(v.frame_ts, s[-1][0]) - 1
    gap = 0
    while i >= 0 and v.frame_ts[i] >= t_end - back and gap <= 4:
        cur = chain[-1][1]
        best, best_iou = None, 0.3
        for (tid2, *e) in v.by_frame[i]:
            if e[5] != ty:
                continue
            ov = _iou(e[1], cur)
            if tid2 == tid:
                ov += 1.0                      # prefer the same track when it's there
            if ov > best_iou:
                best, best_iou = tuple(e), ov
        if best is None:
            gap += 1
        else:
            chain.append(best)
            gap = 0
        i -= 1
    return chain[::-1]


def features(v: Video, c: Cand, ratios: Dict[str, float]) -> Dict[str, float]:
    s = v.tracks.get(c.tid, [])
    if STITCH:
        win = stitched(v, c.tid, c.t)
        s = win or s
    else:
        win = [x for x in s if c.t - 1.0 <= x[0] <= c.t + 1e-6]
    f: Dict[str, float] = {}
    if not win:                                  # detector pass on a track not in the saved frames
        win = [(c.t, [0, 0, 1, 1], 0.0, 1.0, "idle", (c.det or {}).get("gate_type", "?"), 0.0)]
    ty = win[-1][5]
    c.gtype = ty
    areas = np.array([x[2] for x in win])
    ts = np.array([x[0] for x in win])
    end_b = win[-1][1]
    f["dur"] = min(3.0, c.t - s[0][0]) if s else 0.0
    f["n_aligned"] = float(sum(x[4] == "aligned" for x in win))
    f["amax"] = float(areas.max())
    f["aend"] = float(areas[-1])
    last = ts >= c.t - 0.3
    if last.sum() >= 2 and areas[last].min() > 0:
        f["growth"] = float(np.polyfit(ts[last], np.log(areas[last] + 1e-4), 1)[0])
    else:
        f["growth"] = 0.0
    f["cmin"] = float(min(x[3] for x in win))
    f["cend"] = float(win[-1][3])
    f["yolo"] = float(np.mean([x[6] for x in win]))
    for name in ("arch", "flagpole"):
        f["is_" + name] = float(ty == name)
    free = [x for x in win if not any(_borders(v, x[1])) and x[2] >= 0.02]
    if free and ty in ratios:
        r = np.median([(x[1][2] - x[1][0]) / max(1.0, x[1][3] - x[1][1]) for x in free])
        f["aspect"] = float(min(1.5, r / ratios[ty]))      # < 1: seen at an angle
        f["has_aspect"] = 1.0
    else:
        f["aspect"], f["has_aspect"] = 1.0, 0.0
    bl, br, bt, bb = _borders(v, end_b)
    f["borders"] = float(bl + br + bt + bb)
    f["border_side"] = float(bl or br)
    f["border_top"] = float(bt)
    f["shrink_exit"] = float(c.kind == "shrink")
    f["jump_exit"] = float(c.kind == "jump")
    # after: does the same gate go on?
    cont, nxt = 0.0, 1.0
    end_cx, end_cy = (end_b[0] + end_b[2]) / 2 / v.W, (end_b[1] + end_b[3]) / 2 / v.H
    for tid, s2 in v.tracks.items():
        for t, b, a, _, _, ty2, _ in s2:
            if not (c.t < t <= c.t + 0.6):
                continue
            if t <= c.t + 0.3 and ty2 == ty and a >= 0.5 * max(f["aend"], 1e-3):
                cx, cy = (b[0] + b[2]) / 2 / v.W, (b[1] + b[3]) / 2 / v.H
                if _iou(b, end_b) > 0.3 or ((cx - end_cx) ** 2 + (cy - end_cy) ** 2) ** 0.5 < 0.1:
                    cont = max(cont, min(2.0, a / max(f["aend"], 1e-3)))
            if tid != c.tid and a >= BIG:
                nxt = min(nxt, t - c.t)
    f["continues"] = cont
    f["next_big"] = nxt
    f["det"] = float(c.det is not None)
    # camera motion
    med_res = _moving_median(v) if v.two_stage else (float(np.median(v.motion["resid"])) or 1e-3)
    f["exp_before"] = float(_mwin(v, "expand", c.t - 0.5, c.t).mean())
    f["exp_after"] = float(_mwin(v, "expand", c.t, c.t + 0.5).mean())
    f["exp_peak"] = float(_mwin(v, "expand", c.t - 0.2, c.t + 0.2).max())
    f["turn"] = float(abs(_mwin(v, "pan_x", c.t, c.t + 0.5).mean() - _mwin(v, "pan_x", c.t - 0.5, c.t).mean())
                      + abs(_mwin(v, "pan_y", c.t, c.t + 0.5).mean() - _mwin(v, "pan_y", c.t - 0.5, c.t).mean()))
    f["chaos"] = float(_mwin(v, "resid", c.t, c.t + 0.3).max() / med_res)
    return f


def _moving_median(v: Video) -> float:
    """The feed's usual camera-motion noise over frames that move (two stages): a feed gray or still for most of the
    video (a late start, an early crash) would otherwise give ~0, and every real pass would look like chaos."""
    if "_med_res" not in v.motion:
        r = np.asarray(v.motion.get("resid") or [], dtype=float)
        moving = r[r > STILL_RESID]
        v.motion["_med_res"] = float(np.median(moving)) if len(moving) >= 30 else TYPICAL_RESID
    return v.motion["_med_res"]


TRACK_FEATS = ["dur", "n_aligned", "amax", "aend", "growth", "cmin", "cend", "yolo", "is_arch",
               "is_flagpole", "aspect", "has_aspect", "borders", "border_side", "border_top",
               "shrink_exit", "continues", "next_big"]
MOTION_FEATS = ["exp_before", "exp_after", "exp_peak", "turn", "chaos"]


# ──────────────────────────────────────────────────────────────
# Labels and model
# ──────────────────────────────────────────────────────────────

def label(v: Video, cands: List[Cand]) -> np.ndarray:
    """1 for the candidate closest (≤TOL) to each real pass; candidates near
    ignored marks get −1 (left out of training)."""
    y = np.zeros(len(cands))
    real = real_passes(v)
    pairs = sorted((abs(m["t"] - c.t), i, j) for i, m in enumerate(real)
                   for j, c in enumerate(cands) if abs(m["t"] - c.t) <= TOL)
    ur, uc = set(), set()
    for _, i, j in pairs:
        if i not in ur and j not in uc:
            ur.add(i)
            uc.add(j)
            y[j] = 1
    ign = ignored_times(v)
    for j, c in enumerate(cands):
        if y[j] == 0 and any(abs(c.t - t) <= TOL for t in ign):
            y[j] = -1
    return y


class Logistic:
    def __init__(self, names: List[str], l2: float = 1.0):
        self.names, self.l2 = names, l2

    def _X(self, F: List[Dict[str, float]]) -> np.ndarray:
        return np.array([[f[n] for n in self.names] for f in F], dtype=float)

    def fit(self, F: List[Dict[str, float]], y: np.ndarray) -> "Logistic":
        with np.errstate(all="ignore"):          # macOS Accelerate matmul raises spurious FP warnings
            return self._fit(F, y)

    def _fit(self, F: List[Dict[str, float]], y: np.ndarray) -> "Logistic":
        X = self._X(F)
        sd = X.std(0)
        self.mu, self.sd = X.mean(0), np.where(sd < 1e-3, 1.0, sd)   # constant in training: don't scale
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], self.l2)
        reg[0] = 0.0
        for _ in range(50):                       # Newton / IRLS
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (p - y) + reg * w
            Hs = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
            step = np.linalg.solve(Hs, g)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def predict(self, F: List[Dict[str, float]]) -> np.ndarray:
        Z = (self._X(F) - self.mu) / self.sd
        with np.errstate(all="ignore"):
            return 1 / (1 + np.exp(-np.clip(self.w[0] + Z @ self.w[1:], -30, 30)))

    def to_json(self) -> dict:
        return {"kind": "logistic", "names": self.names, "l2": self.l2, "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Logistic":
        m = cls(d["names"], d.get("l2", 1.0))
        m.mu, m.sd, m.w = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"])
        return m


class Stumps:
    """Gradient-boosted decision stumps (logistic loss): picks up "big AND at
    the border"-style rules that a linear model can't."""

    def __init__(self, names: List[str], rounds: int = 150, lr: float = 0.2):
        self.names, self.rounds, self.lr = names, rounds, lr

    def _X(self, F):
        return np.array([[f[n] for n in self.names] for f in F], dtype=float)

    def fit(self, F, y) -> "Stumps":
        X = self._X(F)
        self.cuts = [np.unique(np.percentile(X[:, j], np.arange(5, 100, 5))) for j in range(X.shape[1])]
        pos = min(max(y.mean(), 1e-3), 1 - 1e-3)
        self.base = float(np.log(pos / (1 - pos)))
        Fx = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.rounds):
            p = 1 / (1 + np.exp(-Fx))
            g, h = y - p, np.maximum(p * (1 - p), 1e-6)
            best = None
            for j, cuts in enumerate(self.cuts):
                for cut in cuts:
                    left = X[:, j] <= cut
                    gl, hl = g[left].sum(), h[left].sum()
                    gr, hr = g[~left].sum(), h[~left].sum()
                    gain = gl * gl / (hl + 1) + gr * gr / (hr + 1)
                    if best is None or gain > best[0]:
                        best = (gain, j, cut, gl / (hl + 1), gr / (hr + 1))
            _, j, cut, vl, vr = best
            self.trees.append((j, float(cut), self.lr * vl, self.lr * vr))
            Fx += np.where(X[:, j] <= cut, self.lr * vl, self.lr * vr)
        return self

    def predict(self, F) -> np.ndarray:
        X = self._X(F)
        Fx = np.full(len(X), self.base)
        for j, cut, vl, vr in self.trees:
            Fx += np.where(X[:, j] <= cut, vl, vr)
        return 1 / (1 + np.exp(-Fx))

    def to_json(self) -> dict:
        return {"kind": "stumps", "names": self.names, "base": self.base, "trees": self.trees}

    @classmethod
    def from_json(cls, d: dict) -> "Stumps":
        m = cls(d["names"])
        m.base, m.trees = d["base"], [tuple(t) for t in d["trees"]]
        return m

    def importance(self) -> Dict[str, float]:
        imp: Dict[str, float] = {}
        for j, _, vl, vr in self.trees:
            imp[self.names[j]] = imp.get(self.names[j], 0.0) + abs(vl - vr)
        return imp


# ──────────────────────────────────────────────────────────────
# Decision logics
# ──────────────────────────────────────────────────────────────
#
#   detector         the live detector's passes as they are (baseline)
#   det_continue     detector passes, minus those whose gate is still there
#                    0.3 s later (only removes false passes)
#   rules            hand rules on all candidates: big, approaching, centred
#                    approach and the gate does not continue
#   tracks           logistic model on track features
#   tracks_motion    logistic model on track + camera-motion features
#   tracks_motion_det  … plus "the live detector fired here"
#   tracks_det       track features + "detector fired" (no motion)
#   *_stumps         the same feature sets with boosted stumps instead of
#                    logistic regression

LOGICS = ["detector", "det_continue", "rules",
          "tracks", "tracks_motion", "tracks_det", "tracks_motion_det",
          "tracks_stumps", "tracks_det_stumps", "tracks_motion_det_stumps"]
MODEL_FEATS = {
    "tracks": TRACK_FEATS,
    "tracks_motion": TRACK_FEATS + MOTION_FEATS,
    "tracks_det": TRACK_FEATS + ["det"],
    "tracks_motion_det": TRACK_FEATS + MOTION_FEATS + ["det"],
}
for _k in list(MODEL_FEATS):
    if _k != "tracks_motion":
        MODEL_FEATS[_k + "_stumps"] = MODEL_FEATS[_k]


def make_model(logic: str, l2: float):
    if logic.endswith("_stumps"):
        return Stumps(MODEL_FEATS[logic])
    return Logistic(MODEL_FEATS[logic], l2)


def nms(cands: List[Cand], keep: np.ndarray, window: float = NMS) -> List[Cand]:
    picked: List[Cand] = []
    for c in sorted((c for c, k in zip(cands, keep) if k), key=lambda c: -c.p):
        if all(abs(c.t - o.t) > window for o in picked):
            picked.append(c)
    return sorted(picked, key=lambda c: c.t)


def rule_score(f: Dict[str, float]) -> float:
    ok = (f["amax"] >= 0.2 and f["cmin"] <= 0.3 and f["n_aligned"] >= 2
          and f["continues"] < 0.5 and (f["borders"] >= 1 or f["aend"] >= 0.3))
    return 1.0 if ok else 0.0


def to_pass(c: Cand, fps: float = 30.0) -> dict:
    """Output in race_data pass format (keeps the detector's entry if it fired here).
    A track's last frame is the frame before the gate is gone, so a pass taken
    from a track end is reported one frame later — like the live detector."""
    p = dict(c.det) if c.det else {"gate_id": -1, "gate_type": c.gtype, "sim": 0.0, "source": "NEW",
                                   "reason": f"scorer_{c.kind}", "track_id": c.tid}
    p["t"] = round(c.det["t"] if c.det else c.t + (1.0 / fps if c.kind in ("end", "shrink", "jump") else 0.0), 4)
    p["p_pass"] = round(float(c.p), 4)
    return p


def decide(logic: str, v: Video, cands: List[Cand], model=None, thr: float = 0.5) -> List[dict]:
    if logic == "detector":
        return list(live_passes(v.race))
    if logic == "det_continue":
        out = []
        for c in cands:
            if c.det is None:
                continue
            if c.feats["continues"] < 0.5:
                out.append(dict(c.det))
        return out
    if logic == "rules":
        for c in cands:
            c.p = rule_score(c.feats) + 0.01 * c.feats["amax"]
        return [to_pass(c, v.motion.get("fps", 30.0)) for c in nms(cands, np.array([c.p >= 1 for c in cands]))]
    if not cands:                                        # a pilot who never flew
        return []
    ps = model.predict([c.feats for c in cands])
    for c, p in zip(cands, ps):
        c.p = float(p)
    return [to_pass(c, v.motion.get("fps", 30.0))
            for c in nms(cands, ps >= thr, NMS_TWO_STAGE if v.two_stage else NMS)]


def best_threshold(videos, cands_by, model) -> float:
    """Threshold that minimises missed + false + off-time on the training videos."""
    best, best_err = 0.5, None
    for thr in np.arange(0.2, 0.81, 0.05):
        err = 0
        for v in videos:
            err += errors(v, decide("m", v, cands_by[v.stem], model, thr))
        if best_err is None or err < best_err:
            best, best_err = round(float(thr), 2), err
    return best


def errors(v: Video, passes: List[dict]) -> int:
    from score import match_by_time
    real = real_passes(v)
    ign = ignored_times(v)
    preds = [p["t"] for p in passes if not any(abs(p["t"] - t) <= TOL for t in ign)]
    on = match_by_time([m["t"] for m in real], preds, TOL)
    return (len(real) - len(on)) + (len(preds) - len(on))


# ──────────────────────────────────────────────────────────────
# Leave-one-video-out
# ──────────────────────────────────────────────────────────────

def prepare(tp: TrackPaths, include_incomplete: bool = True) -> Tuple[List[Video], Dict[str, List[Cand]]]:
    videos = []
    for g in sorted(tp.gt_dir.glob("*.gt.json")):
        stem = g.name[: -len(".gt.json")]
        gt = json.loads(g.read_text(encoding="utf-8"))
        if gt.get("status") != "complete" and not include_incomplete:
            continue
        if tp.race_data(stem).exists():
            videos.append(load_video(tp, stem))
    ratios = type_ratios(videos)
    cands_by = {}
    for v in videos:
        cs = candidates(v)
        for c in cs:
            c.feats = features(v, c, ratios)
        cands_by[v.stem] = cs
    return videos, cands_by


def lovo_passes(videos: List[Video], cands_by: Dict[str, List[Cand]], logic: str,
                l2: float = 1.0) -> Dict[str, List[dict]]:
    """Each video's passes from a model trained on the other videos."""
    out = {}
    for v in videos:
        model, thr = None, 0.5
        if logic in MODEL_FEATS:
            train = [o for o in videos if o.stem != v.stem]
            F, Y = [], []
            for o in train:
                y = label(o, cands_by[o.stem])
                for c, yy in zip(cands_by[o.stem], y):
                    if yy >= 0:
                        F.append(c.feats)
                        Y.append(yy)
            model = make_model(logic, l2).fit(F, np.array(Y))
            thr = best_threshold(train, cands_by, model)
        out[v.stem] = decide(logic, v, cands_by[v.stem], model, thr)
    return out


def train_all(videos, cands_by, logic: str, l2: float = 1.0):
    F, Y = [], []
    for v in videos:
        for c, yy in zip(cands_by[v.stem], label(v, cands_by[v.stem])):
            if yy >= 0:
                F.append(c.feats)
                Y.append(yy)
    model = make_model(logic, l2).fit(F, np.array(Y))
    return model, best_threshold(videos, cands_by, model)


# ──────────────────────────────────────────────────────────────
# Deciding a new video with the saved model (used by extract_race.py)
# ──────────────────────────────────────────────────────────────

def model_path() -> Path:
    """The pass scorer of the model picked (model_profiles.py): v2 <data_root>/training/pass_scorer.json, v3 the
    one in the repo (cozYnut's, two stages)."""
    import model_profiles
    return model_profiles.scorer_path()


def load_model(path: Optional[Path] = None) -> Optional[dict]:
    path = path or model_path()
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d["model"]
    d["_model"] = Stumps.from_json(m) if m.get("kind") == "stumps" else Logistic.from_json(m)
    return d


def decide_race(race: dict, motion: dict, W: int, H: int, saved: dict, stem: str = "") -> List[dict]:
    """Passes for one video (race_data with its live passes + motion) decided by
    the saved model. Returns race_data-format passes, each with p_pass."""
    v = build_video(stem, race, motion, W, H)
    v.two_stage = "decoder" in saved
    cands = candidates(v)
    ratios = saved.get("type_ratios", {})
    for c in cands:
        c.feats = features(v, c, ratios)
    if saved.get("logic") in ("detector", "det_continue", "rules"):
        return decide(saved["logic"], v, cands)
    return decide("model", v, cands, saved["_model"], float(saved.get("threshold", 0.5)))


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    import score
    ap = argparse.ArgumentParser(description="Compare pass decision logics (leave-one-video-out)")
    ap.add_argument("tracks", nargs="*", help="Track folder names (default: all)")
    ap.add_argument("--logic", choices=LOGICS, help="Only this logic (default: compare all)")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 regularisation of the learned logics")
    ap.add_argument("--list", action="store_true", help="List remaining misses / false / off-time passes")
    ap.add_argument("--save", action="store_true",
                    help="Train --logic (default BEST_LOGIC) on all videos → training/pass_scorer.json")
    args = ap.parse_args()

    tracks = [tp for tp in list_tracks() if not args.tracks or tp.dir.name in args.tracks]
    sargs = argparse.Namespace(tol=TOL, off_tol=0.5, rematch=False, include_incomplete=True, list=args.list,
                               sim_thresh=0.88, min_margin=0.03, g1_sim_thresh=None, g1_margin=None,
                               require_same_type=False)
    for tp in tracks:
        if not tp.gt_dir.exists():
            continue
        videos, cands_by = prepare(tp)
        if not videos:
            continue
        n_real = sum(len(real_passes(v)) for v in videos)
        covered = 0
        for v in videos:
            covered += sum(y == 1 for y in label(v, cands_by[v.stem]))
        print(f"\n{tp.name}: {len(videos)} videos, {n_real} real passes, "
              f"{sum(len(c) for c in cands_by.values())} candidates, candidate recall {covered}/{n_real}")
        if args.save:
            logic = args.logic or BEST_LOGIC
            model, thr = train_all(videos, cands_by, logic, args.l2)
            out = model_path()
            out.parent.mkdir(parents=True, exist_ok=True)
            det_models = sorted({(v.gt or {}).get("det_model") or v.race.get("det_model") or "?" for v in videos})
            out.write_text(json.dumps({"logic": logic, "threshold": thr, "model": model.to_json(),
                                       "type_ratios": type_ratios(videos), "det_models": det_models,
                                       "trained_on": [v.stem for v in videos]}, indent=2), encoding="utf-8")
            print(f"Saved {logic} (threshold {thr:.2f}) → {out}")
            weights = (model.importance() if isinstance(model, Stumps)
                       else dict(zip(model.names, model.w[1:])))
            for n, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
                print(f"    {n:<12} {w:+.2f}")
            continue
        rows = []
        for logic in ([args.logic] if args.logic else LOGICS):
            per = lovo_passes(videos, cands_by, logic, args.l2)
            vs = [score.score_video(tp, tp.gt(v.stem), sargs, passes=per[v.stem]) for v in videos]
            t = score.total(vs)
            fired = t["found"] + t["off_time"]
            rows.append((logic, t["found"], t["off_time"], t["missed"], t["false"], t["real"],
                         fired / max(1, fired + t["false"])))
            if args.list:
                print(f"\n  {logic}")
                for vv in vs:
                    score.print_video(vv, sargs)
        print(f"\n  {'logic':<26}{'on time':>9}{'off-time':>10}{'missed':>8}{'false':>7}{'errors':>8}{'precision':>11}")
        for logic, on, off, miss, fp, real, prec in rows:
            print(f"  {logic:<26}{on:>5}/{real:<3}{off:>10}{miss:>8}{fp:>7}{off + miss + fp:>8}{prec:>10.0%}")


if __name__ == "__main__":
    main()
