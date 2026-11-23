"""
The race view (static/race/, the dashboard's Race view tab): a heat played back as a race, in floating windows —
the heat's 2×2 video with each pilot's tile, a timeline lane per pilot, the standings at the playhead, a race map
and the race in 3D; or one pilot's run (its quadrant of the video, laps, gate splits, missed gates). From
gilboash/therealspotter's web app (web/static/app.js, Race mode).

Read-only, from what the dashboard already reads (race_day.json, runs/<run>.race_data.json and .stats.json),
plus where the gates roughly are, for the race map and the 3D view (layout() below).

    /api/race/heats       every heat with an analysed run: its label, pilots, class, a pilot with a missed gate
    /api/race/heat/<n>    one heat for the race view: its video, each pilot's passes, laps (FPVTrackside's when the
                          event has timing, as the results show them) and stats, the track, its layout
"""

import glob
import json
import math
import statistics
import threading
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dataset_paths import TrackPaths

SPAN_PAD = 4.0          # s: the timeline shows the race (the passes of its pilots) and this much around it
LAYOUT_RUNS = 16        # runs read (the newest) to estimate a layout: plenty of clean laps, a few seconds


def _n_gates(tp: TrackPaths, stats: List[dict]) -> int:
    try:
        return int(json.loads(tp.learned_timing.read_text(encoding="utf-8"))["n_gates"])
    except (OSError, ValueError, KeyError):
        ns = [s["n_gates"] for s in stats if s and s.get("n_gates")]
        return max(set(ns), key=ns.count) if ns else 0


def _legs(tp: TrackPaths, n: int) -> List[float]:
    """The learned time of every leg G1→G2 … Gn→G1 (seconds), or [] when the track isn't learned."""
    try:
        ls = json.loads(tp.learned_timing.read_text(encoding="utf-8")).get("leg_seconds", {})
    except (OSError, ValueError):
        return []
    legs = [float(ls.get(f"G{g}->G{g % n + 1}") or 0) for g in range(1, n + 1)]
    return legs if all(legs) else []


def heats(v) -> List[dict]:
    """The heats the race view can play, in the order flown (the results' heats with an analysed run)."""
    stats = {s["stem"]: s for s in v.all_stats()}
    out = []
    for h in v.heats(list(stats.values())):
        if h["heat"] is None:
            continue
        ps = [p for p in h["pilots"] if p.get("stem")]
        check = [p["pilot"] for p in ps if (stats.get(p["stem"]) or {}).get("missed_gates")
                 and (stats.get(p["stem"]) or {}).get("reviewed") != "complete"]
        out.append({"heat": h["heat"], "label": h["label"] or f"Heat {h['heat']}", "start": h["start"],
                    "layout": h["layout"], "pilots": [p["pilot"] for p in ps], "check": check})
    return out


def heat(v, n: int) -> Optional[dict]:
    runs = sorted((r for r in v.visible_runs() if r["heat"] == n and r.get("status") == "labelled"),
                  key=lambda r: (r.get("quad") is None, r.get("quad") or 0, r["stem"]))
    if not runs:
        return None
    tp = v._tpx(runs[0]["stem"])
    stats = [v._json(v._tpx(r["stem"]).runs_dir / f"{r['stem']}.stats.json") for r in runs]
    n_gates = _n_gates(tp, stats)
    pilots, videos = [], set()
    for k, (r, st) in enumerate(zip(runs, stats)):
        stem = r["stem"]
        race = v._json(v._tpx(stem).race_data(stem)) or {}
        w0, w1 = r.get("window") or [float("-inf"), float("inf")]
        passes = sorted(({"t": round(p["t"], 3), "gate_id": p["gate_id"], "recovered": bool(p.get("recovered"))}
                         for p in race.get("passes", []) if p.get("gate_id", -1) >= 1 and w0 <= p["t"] <= w1),
                        key=lambda p: p["t"])
        # the laps as the results have them: FPVTrackside's when the event has timing (stats), else the video's
        if st and st.get("lap_marks") is not None:
            laps = [{"lap": m["lap"], "t0": m["t0"], "t1": m["t1"], "dt": round(m["t1"] - m["t0"], 3)}
                    for m in st["lap_marks"]]
        else:
            laps = [{"lap": L["lap"], "t0": L["t0"], "t1": L["t1"], "dt": L["dt"]} for L in race.get("laps", [])
                    if L["t0"] >= w0 and L["t1"] <= w1]
        for L in laps:                     # the gates seen in each lap (what's missing: a gate to check)
            L["splits"] = [{"gate_id": p["gate_id"], "t": p["t"], "recovered": p["recovered"]} for p in passes
                           if L["t0"] - 0.05 <= p["t"] < L["t1"] - 0.05]
        vf = v.video_file(stem)
        videos.add(str(Path(vf).resolve()))
        pilots.append({"i": k, "stem": stem, "name": r.get("pilot") or r.get("channel") or f"Pilot {k + 1}",
                       "channel": r.get("channel"), "crop": r.get("crop"), "video_url": f"/video/{stem}.mp4{v.q}",
                       "live_start_s": r.get("live_start_s") or 0.0, "fps": race.get("fps"), "passes": passes, "laps": laps,
                       "end": passes[-1]["t"] if passes else None, "stats": st, "reviewed": r.get("reviewed")})
    ts = [p["t"] for q in pilots for p in q["passes"]] + [r["live_start_s"] for r in runs if r.get("live_start_s")]
    lay = v.layout_of(runs[0]["stem"])
    field = v.field(lay)
    info = next((h for h in v.heats(v.all_stats()) if h["heat"] == n), None)
    return {"heat": n, "label": (info or {}).get("label") or f"Heat {n}", "start": (info or {}).get("start"),
            "class": lay, "n_gates": n_gates, "legs": _legs(tp, n_gates) if n_gates else [],
            "layout": layout(tp, n_gates), "gates": gate_pictures(v, tp),
            "field": {k: field.get(k) for k in ("median_lap", "section_median", "sections", "n_gates")} if field else None,
            # one video for every pilot, each in its part of the picture: the race view plays them together
            "combo": len(pilots) > 1 and len(videos) == 1 and all(p["crop"] for p in pilots),
            "video_url": pilots[0]["video_url"],
            "span": [max(0.0, min(ts) - SPAN_PAD), max(ts) + SPAN_PAD] if ts else None,
            "pilots": pilots}


def gate_pictures(v, tp: TrackPaths) -> List[dict]:
    """The learned gates with a few of their pictures (as the Track page shows them), for the Gates window."""
    from race_day.trackside import relocate
    mem = v._json(tp.gate_memory)
    if not mem:
        return []
    root = Path(v.tp.dir).resolve()
    amp = "&" + v.q[1:] if v.q else ""
    out = []
    for g in sorted(mem.get("memory", []), key=lambda g: g.get("order_idx", g["gate_id"])):
        found = [relocate(p, tp.dir) for p in g.get("embed_imgs", []) if p]
        imgs = [f.resolve() for f in found if f is not None and str(f.resolve()).startswith(str(root))][:6]
        out.append({"id": g["gate_id"], "type": g.get("gate_type", "?"), "offset_s": g.get("offset_s"),
                    "images": [f"/img?p={urllib.parse.quote(str(f.relative_to(root)))}{amp}" for f in imgs]})
    return out


# ── where the gates roughly are: a top view for the race map and the 3D view ──────────────────────────────────────
# Nothing measures the field, so it's estimated from what the analysed runs already have (gilboash/therealspotter's
# learn_layout.py): how the camera moved (runs/<run>.motion.json) between two gates of a clean lap (every gate
# passed, in order). A leg's length: its learned time (speed taken as even); which way the pilot turned on it: the
# camera's roll (an FPV drone banks into a turn); how much: the camera's pitch (banked, a turn shows mostly as
# pitch). The median per leg over the clean laps, then the path is closed. A rough shape: a layout made (or moved by
# hand) in therealspotter's Track layout window can be put in the track folder as gate_layout.json, and is used as
# it is.

_layouts: Dict[str, tuple] = {}
_layout_lock = threading.Lock()


def layout(tp: TrackPaths, n: int) -> Optional[dict]:
    """{n_gates, gates: [{id, x, y, type}], method, laps_used}: x, y in seconds of flight, y down; None: not learned."""
    if n < 2 or not tp.learned_timing.exists():
        return None
    kind = {}
    try:
        kind = {g["gate_id"]: g.get("gate_type", "?") for g in
                json.loads(tp.gate_memory.read_text(encoding="utf-8")).get("memory", [])}
    except (OSError, ValueError):
        pass
    made = tp.dir / "gate_layout.json"
    d = None
    if made.exists():
        try:
            d = json.loads(made.read_text(encoding="utf-8"))
        except ValueError:
            d = None
        if d and d.get("n_gates") != n:
            d = None
        elif d:
            d = dict(d, method=d.get("method", "") + " (gate_layout.json)")
    if d is None:
        files = sorted(glob.glob(str(tp.runs_dir / "*.motion.json")), key=lambda f: -Path(f).stat().st_mtime)
        key = (tp.learned_timing.stat().st_mtime, len(files), Path(files[0]).stat().st_mtime if files else 0)
        with _layout_lock:
            hit = _layouts.get(str(tp.dir))
            if not hit or hit[0] != key:
                hit = (key, _estimate(tp, n, files[:LAYOUT_RUNS]))
                _layouts[str(tp.dir)] = hit
        d = hit[1]
    man = d.get("manual", {})
    gates = [{"id": i + 1, "x": (man.get(str(i + 1)) or p)[0], "y": (man.get(str(i + 1)) or p)[1],
              "type": kind.get(i + 1, "?")} for i, p in enumerate(d["estimated"])]
    return {"n_gates": n, "gates": gates, "method": d.get("method"), "laps_used": d.get("laps_used")}


def _estimate(tp: TrackPaths, n: int, motion_files: List[str]) -> dict:
    learned = _legs(tp, n) or [0.0] * n
    legs, laps, how = _clean_legs(n, motion_files, learned)
    if laps >= 2 and all(legs):
        ln = [learned[i] or statistics.median(e["dt"] for e in legs[i]) for i in range(n)]
        th = np.array([statistics.median(e["turn"] * np.sign(e["roll"]) for e in legs[i]) for i in range(n)])
        th = th * (2 * math.pi / th.sum()) if abs(th.sum()) > 1e-6 else np.full(n, 2 * math.pi / n)
        method = f"camera turn per leg, {laps} {'clean ' if how == 'clean' else ''}laps; leg lengths: learned times"
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
    pts = (pts - np.outer(np.arange(n + 1) / n, err))[:n]      # closed: the error spread over the lap
    pts = pts - pts.mean(axis=0)
    pts[:, 1] *= -1                                   # y down, as drawn
    return {"n_gates": n, "estimated": [[round(float(x), 3), round(float(y), 3)] for x, y in pts],
            "laps_used": laps, "method": method}


def _clean_legs(n: int, motion_files: List[str], learned: List[float]) -> tuple:
    """Per leg, over every clean lap: its time, turn and roll ("clean"). Fewer than 2 clean laps (a gate the detector
    mostly misses): each leg over every lap that saw both its gates, and a leg no lap saw both ends of gets its share,
    by the learned times, of a leg over that one missing gate ("per leg")."""
    clean, n_clean = [[] for _ in range(n)], 0
    legs, laps = [[] for _ in range(n)], 0
    over = [[] for _ in range(n)]                     # leg i and i+1 together (gate i+2 missed)
    for mf in motion_files:
        rf = Path(mf[: -len(".motion.json")] + ".race_data.json")
        if not rf.exists():
            continue
        try:
            rd, m = json.loads(rf.read_text(encoding="utf-8")), json.loads(Path(mf).read_text(encoding="utf-8"))
            fps, py, rl = m["fps"], np.array(m["pan_y"]), np.array(m["roll"])
        except (OSError, ValueError, KeyError):
            continue
        bias = float(np.median(py)) if len(py) else 0.0
        for L in rd.get("laps", []):
            sp = L.get("splits", [])
            first = {}
            for s in sorted(sp, key=lambda s: s["t"]):      # a gate named twice: its first pass
                if not s.get("recovered"):
                    first.setdefault(s["gate_id"], s["t"])
            if 1 not in first:
                continue
            ts = [first.get(g) for g in range(1, n + 1)] + [L["t1"]]     # gate n+1: the next lap's G1

            def leg(t0, t1):
                a, b = int(t0 * fps), int(t1 * fps)
                return {"dt": t1 - t0, "turn": float((py[a:b] - bias).sum()), "roll": float(rl[a:b].sum())}
            if not any(s.get("recovered") for s in sp) and sorted(first) == list(range(1, n + 1)) \
                    and all(b > a for a, b in zip(ts, ts[1:])):
                n_clean += 1
                for i in range(n):
                    clean[i].append(leg(ts[i], ts[i + 1]))
            used = False
            for i in range(n):
                if ts[i] is not None and ts[i + 1] is not None and ts[i + 1] > ts[i]:
                    legs[i].append(leg(ts[i], ts[i + 1]))
                    used = True
                elif i + 2 <= n and ts[i] is not None and ts[i + 1] is None and ts[i + 2] is not None \
                        and ts[i + 2] > ts[i]:
                    over[i].append(leg(ts[i], ts[i + 2]))
            laps += used
    for i in range(n):                                # a leg never seen whole: its share of one over its far end
        for k in (i, i - 1):                          # (over[i]: legs i, i+1) or its near end (over[i-1]: i-1, i)
            if legs[i] or k < 0 or not over[k]:
                continue
            w = learned[i] / (learned[k] + learned[k + 1]) if learned[k] and learned[k + 1] else 0.5
            legs[i] = [{key: e[key] * w for key in ("dt", "turn", "roll")} for e in over[k]]
    return (clean, n_clean, "clean") if n_clean >= 2 else (legs, laps, "per leg")
