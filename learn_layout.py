#!/usr/bin/env python3
"""
learn_layout.py — where the gates of a learned track roughly are (a top view), after learning.

Nothing measures the field, so it's estimated from what every analysed video already has: how the camera moved
(runs*/<video>.motion.json) between two gates of a clean lap (every gate passed, in order; a gate named twice counts once).
  - a leg's length: the learned time between its two gates (speed taken as even);
  - which way the pilot turned on it: the camera's roll (an FPV drone banks into a turn) — the same way on 14–16 of
    16 laps of fril-27 on most legs;
  - how much: the camera's pitch (pan_y less its straight-flight bias): banked, a turn shows mostly as pitch, either
    way. Scaled so a lap turns once around.
Median per leg over every clean lap, then the path is closed (it ends ~40% of a lap from its start before that on
fril-27 — a rough shape: the gates can be dragged where they really are in the web app's Track layout window).

<track>/gate_layout.json (v3: gate_layout.v3.json, like every learned file; a model without one uses v2's):
  estimated  [[x, y] per gate], in seconds of flight, y down (as drawn)
  manual     {gate id: [x, y]} — gates moved by hand; kept when it's estimated again
  view       {mirror, turns}: mirrored / turned by hand — every new estimate is too
  laps_used, closure (how far the path ended from its start, as a share of the lap), method

    python learn_layout.py <track>          # estimate (keeps the gates moved by hand)
"""

import argparse
import glob
import json
import math
import statistics as st
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from dataset_paths import TrackPaths, list_tracks


def _legs(tp: TrackPaths) -> tuple:
    d = json.loads(tp.learned_timing.read_text(encoding="utf-8"))
    n, ls = int(d["n_gates"]), d.get("leg_seconds", {})
    return n, [float(ls.get(f"G{i}->G{i % n + 1}") or 0) for i in range(1, n + 1)]


def _clean_legs(tp: TrackPaths, n: int) -> tuple:
    """Per leg, over every clean lap of every analysed video (any model's): its time, turn and roll."""
    legs, laps, seen = [[] for _ in range(n)], 0, set()
    for f in sorted(glob.glob(str(tp.dir / "runs*" / "*.race_data.json"))):
        stem = Path(f).name[: -len(".race_data.json")]
        mf = Path(f).parent / f"{stem}.motion.json"
        if stem in seen or not mf.exists():
            continue                                  # one analysis per video (motion is the same for every model)
        seen.add(stem)
        rd, m = json.loads(Path(f).read_text(encoding="utf-8")), json.loads(mf.read_text(encoding="utf-8"))
        fps, py, rl = m["fps"], np.array(m["pan_y"]), np.array(m["roll"])
        bias = float(np.median(py))
        for L in rd.get("laps", []):
            sp = L.get("splits", [])
            first = {}
            for s in sorted(sp, key=lambda s: s["t"]):      # a gate named twice: its first pass
                first.setdefault(s["gate_id"], s["t"])
            if any(s.get("recovered") for s in sp) or sorted(first) != list(range(1, n + 1)):
                continue
            ts = [first[g] for g in range(1, n + 1)] + [L["t1"]]
            if any(b <= a for a, b in zip(ts, ts[1:])):
                continue
            laps += 1
            for i in range(n):
                a, b = int(ts[i] * fps), int(ts[i + 1] * fps)
                legs[i].append({"dt": ts[i + 1] - ts[i], "turn": float((py[a:b] - bias).sum()), "roll": float(rl[a:b].sum())})
    return legs, laps


def estimate(tp: TrackPaths) -> dict:
    n, learned = _legs(tp)
    legs, laps = _clean_legs(tp, n)
    if laps >= 2:
        ln = [learned[i] or st.median(e["dt"] for e in legs[i]) for i in range(n)]
        th = np.array([st.median(e["turn"] * np.sign(e["roll"]) for e in legs[i]) for i in range(n)])
        th = th * (2 * math.pi / th.sum()) if abs(th.sum()) > 1e-6 else np.full(n, 2 * math.pi / n)
        method = f"camera turn per leg (pitch × roll's side), {laps} clean laps; leg lengths: learned times"
    else:                                             # nothing to go by: a circle, legs by their times
        ln = [x or 1.0 for x in learned]
        th = np.full(n, 2 * math.pi / n)
        method = "a circle (no clean lap analysed yet)"
    h, p, pts = 0.0, np.zeros(2), [np.zeros(2)]
    for i in range(n):                                # half the turn before the leg, half after
        h += th[i] / 2
        p = p + ln[i] * np.array([math.cos(h), math.sin(h)])
        h += th[i] / 2
        pts.append(p.copy())
    pts = np.array(pts)
    err = pts[-1] - pts[0]
    closure = float(np.linalg.norm(err) / sum(ln))
    pts = (pts - np.outer(np.arange(n + 1) / n, err))[:n]
    pts = pts - pts.mean(axis=0)
    pts[:, 1] *= -1                                   # y down, as drawn
    return {"n_gates": n, "estimated": [[round(float(x), 3), round(float(y), 3)] for x, y in pts],
            "laps_used": laps, "closure": round(closure, 3), "method": method, "made": time.strftime("%Y-%m-%d %H:%M")}


_TF = {"mirror": lambda x, y: [-x, y], "rotate": lambda x, y: [-y, x]}       # rotate: a quarter turn clockwise (y down)


def _view(p, view: dict) -> list:
    if view.get("mirror"):
        p = _TF["mirror"](*p)
    for _ in range(view.get("turns", 0) % 4):
        p = _TF["rotate"](*p)
    return p


def save_estimate(tp: TrackPaths) -> dict:
    """Estimate it again, mirrored / turned as before; the gates moved by hand stay where they were put (unless the
    track has other gates now)."""
    old = load_file(tp)
    same = bool(old) and old.get("n_gates") == estimate_n(tp)
    d = estimate(tp)
    d["view"] = old.get("view", {}) if same else {}
    d["estimated"] = [_view(p, d["view"]) for p in d["estimated"]]
    d["manual"] = old.get("manual", {}) if same else {}
    tp.gate_layout_own.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d


def estimate_n(tp: TrackPaths) -> int:
    return _legs(tp)[0]


def load_file(tp: TrackPaths) -> Optional[dict]:
    f = tp.gate_layout
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def gates(d: dict) -> list:
    """Each gate's place: moved by hand, else estimated."""
    man = d.get("manual", {})
    return [{"id": i + 1, "x": (man.get(str(i + 1)) or p)[0], "y": (man.get(str(i + 1)) or p)[1],
             "manual": str(i + 1) in man} for i, p in enumerate(d["estimated"])]


def transform(tp: TrackPaths, how: str) -> dict:
    """mirror (left ↔ right) or rotate (a quarter turn clockwise) the whole layout, the gates moved by hand too."""
    d = load_file(tp)
    f, v = _TF[how], d.setdefault("view", {})
    if how == "mirror":                               # mirror after the turns = the turns the other way, then mirror
        v["mirror"], v["turns"] = not v.get("mirror"), (-v.get("turns", 0)) % 4
    else:
        v["turns"] = (v.get("turns", 0) + 1) % 4
    d["estimated"] = [f(*p) for p in d["estimated"]]
    d["manual"] = {k: f(*p) for k, p in d.get("manual", {}).items()}
    tp.gate_layout_own.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track")
    a = ap.parse_args()
    tp = next((t for t in list_tracks() if t.dir.name == a.track), None)
    if tp is None or not tp.learned_timing.exists():
        sys.exit(f"{a.track}: no such track, or not learned yet")
    d = save_estimate(tp)
    print(f"{tp.gate_layout_own}: {d['n_gates']} gates · {d['method']} · path ended {d['closure']:.0%} of a lap "
          f"from its start (closed) · {len(d['manual'])} moved by hand")


if __name__ == "__main__":
    main()
