"""
Race-day statistics.

run_stats(race, meta, n)  one run: laps, lap times, consistency, section
                          (gate-to-gate leg) times, theoretical best,
                          holeshot, missed gates, run ended mid-lap
field_stats(all_stats)    across the track's runs: per-section median and
                          best, lap median — the "field" each run and pilot
                          is compared with
"""

from statistics import median
from typing import Dict, List, Optional

import numpy as np


def _sections(n: int) -> List[str]:
    return [f"G{g}→G{g % n + 1}" for g in range(1, n + 1)]


def run_stats(race: dict, meta: dict, n: int, window=None, official: Optional[List[float]] = None) -> dict:
    """window = [t0, t1] (video seconds): only that part counts — a video that
    also covers a restarted (void) attempt of the heat.
    official: the timing system's lap crossings (FPVTrackside's current laps,
    holeshot first), in video seconds. Then laps, lap times, holeshot and the
    lap marks are those (the official result); gates and sections still come
    from the video, grouped by these laps."""
    passes = sorted([p for p in race.get("passes", []) if p.get("gate_id", -1) >= 1], key=lambda p: p["t"])
    laps = race.get("laps", [])
    if official is not None:
        laps = [{"lap": i + 1, "t0": a, "t1": b, "dt": b - a} for i, (a, b) in enumerate(zip(official, official[1:]))]
    if window:
        w0, w1 = window
        passes = [p for p in passes if w0 <= p["t"] <= w1]
        laps = [l for l in laps if l["t0"] >= w0 and l["t1"] <= w1]
    lap_times = [round(l["dt"], 3) for l in laps]
    # every leg flown gate g → g+1 (consecutive passes), with the lap it belongs to
    lap_of = lambda t: next((i for i, l in enumerate(laps) if l["t0"] <= t < l["t1"]), None)
    legs: Dict[int, List[float]] = {g: [] for g in range(1, n + 1)}
    grid = [[None] * n for _ in laps]                 # laps × sections
    missed = []
    for a, b in zip(passes, passes[1:]):
        step = (b["gate_id"] - a["gate_id"]) % n
        if step == 1:
            dt = round(b["t"] - a["t"], 3)
            legs[a["gate_id"]].append(dt)
            li = lap_of(a["t"])
            if li is not None:
                grid[li][a["gate_id"] - 1] = dt
        elif step > 1:
            missed += [{"gate": (a["gate_id"] + k - 1) % n + 1, "t": round(a["t"], 2)} for k in range(1, step)]
    best_leg = {g: min(v) for g, v in legs.items() if v}
    live_start = max(float(meta.get("live_start_s", 0.0)), window[0] if window else 0.0)
    last_lap_end = laps[-1]["t1"] if laps else None
    ended_mid_lap = bool(laps) and any(p["t"] > last_lap_end + 0.05 for p in passes) and passes[-1]["gate_id"] != 1
    best3 = None
    if len(lap_times) >= 3:
        best3 = round(min(sum(lap_times[i:i + 3]) for i in range(len(lap_times) - 2)), 3)
    return {
        "n_gates": n,
        "sections": _sections(n),
        "passes": len(passes),
        "laps": len(laps),
        "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else None,
        "best_lap_index": int(np.argmin(lap_times)) if lap_times else None,
        "mean_lap": round(float(np.mean(lap_times)), 3) if lap_times else None,
        "median_lap": round(float(np.median(lap_times)), 3) if lap_times else None,
        "lap_sd": round(float(np.std(lap_times)), 3) if len(lap_times) >= 2 else None,
        "lap_cv": round(float(np.std(lap_times) / np.mean(lap_times)), 4) if len(lap_times) >= 2 else None,
        "best3": best3,
        "section_best": [best_leg.get(g) for g in range(1, n + 1)],
        "section_median": [round(float(np.median(legs[g])), 3) if legs[g] else None for g in range(1, n + 1)],
        "section_grid": grid,
        "theoretical_best": round(sum(best_leg.values()), 3) if len(best_leg) == n else None,
        "holeshot": (round(official[0] - live_start, 2) if official else None) if official is not None
                    else (round(passes[0]["t"] - live_start, 2) if passes else None),
        "laps_from": "timing" if official is not None else "video",
        "lap_marks": [{"lap": l.get("lap", i + 1), "t0": round(l["t0"], 3), "t1": round(l["t1"], 3)}
                      for i, l in enumerate(laps)],
        "first_gate": passes[0]["gate_id"] if passes else None,
        "flight_time": round(passes[-1]["t"] - passes[0]["t"], 2) if len(passes) >= 2 else None,
        "missed_gates": missed,
        "ended_mid_lap": ended_mid_lap,
    }


def field_stats(all_stats: List[dict]) -> dict:
    """The field: per-section median and best over every run's legs, lap median."""
    if not all_stats:
        return {}
    n = all_stats[0]["n_gates"]
    sec_all = [[] for _ in range(n)]
    laps = []
    best = [None] * n
    for s in all_stats:
        laps += s["lap_times"]
        for row in s["section_grid"]:
            for i, v in enumerate(row):
                if v is not None:
                    sec_all[i].append(v)
        for i, v in enumerate(s["section_best"]):
            if v is not None and (best[i] is None or v < best[i]["time"]):
                best[i] = {"time": v, "run": s.get("stem"), "pilot": s.get("pilot")}
    best_lap = min(((x, s) for s in all_stats for x in s["lap_times"]), default=None, key=lambda z: z[0])
    return {
        "n_gates": n,
        "sections": all_stats[0]["sections"],
        "median_lap": round(median(laps), 3) if laps else None,
        "best_lap": {"time": best_lap[0], "run": best_lap[1].get("stem"), "pilot": best_lap[1].get("pilot")}
                    if best_lap else None,
        "section_median": [round(median(v), 3) if v else None for v in sec_all],
        "section_best": best,
        "section_spread": [round(float(np.percentile(v, 90) - np.percentile(v, 10)), 3) if len(v) >= 3 else None
                           for v in sec_all],
        "runs": len(all_stats),
    }


def compare(run: dict, field: dict) -> dict:
    """How a run compares with the field: speed index (field median lap ÷
    this run's median lap, > 1 = faster) and per-section differences."""
    out = {"speed_index": None, "section_vs_median": [], "section_vs_best": []}
    if run.get("median_lap") and field.get("median_lap"):
        out["speed_index"] = round(field["median_lap"] / run["median_lap"], 3)
    for i in range(run["n_gates"]):
        mine = run["section_best"][i]
        med = field["section_median"][i] if field.get("section_median") else None
        bst = (field.get("section_best") or [None] * run["n_gates"])[i]
        out["section_vs_median"].append(round(mine - med, 3) if mine is not None and med else None)
        out["section_vs_best"].append(round(mine - bst["time"], 3) if mine is not None and bst else None)
    return out


# ── the event's results pages: per pilot, per named section ────────────────

def best_in_a_row(laps: List[float], k: int) -> Optional[float]:
    """The best k laps in a row (their total), or None with fewer laps."""
    if k < 1 or len(laps) < k:
        return None
    return round(min(sum(laps[i:i + k]) for i in range(len(laps) - k + 1)), 3)


def named_times(grid: List[list], legs: List[int]) -> List[float]:
    """Every time a run flew a named section (its legs, 0-based, in flying order) in full. A section that
    runs over the start/finish takes its last legs from the next lap's row."""
    cut = next((k for k in range(1, len(legs)) if legs[k] < legs[k - 1]), len(legs))
    a, b = legs[:cut], legs[cut:]
    out = []
    for i, row in enumerate(grid):
        vals = [row[j] for j in a]
        if b:
            if i + 1 >= len(grid):
                continue
            vals += [grid[i + 1][j] for j in b]
        if vals and all(v is not None for v in vals):
            out.append(round(sum(vals), 3))
    return out


def named_field(all_stats: List[dict], named: List[dict]) -> List[dict]:
    """The field in each named section ([{name, from, to, legs}]): its median and who flew it fastest."""
    out = []
    for x in named:
        every, best = [], None
        for s in all_stats:
            ts = named_times(s["section_grid"], x["legs"])
            every += ts
            if ts and (best is None or min(ts) < best["time"]):
                best = {"time": min(ts), "run": s.get("stem"), "pilot": s.get("pilot")}
        out.append({"name": x["name"], "from": x["from"], "to": x["to"],
                    "median": round(median(every), 3) if every else None, "best": best})
    return out


def pilot_summary(pilot: str, runs: List[dict], field: dict, pb: int, named: List[dict]) -> dict:
    """One pilot over the whole event (their runs on one layout): the leaderboard's row, and what another
    pilot is compared with (best and typical lap, best and typical time in every section)."""
    n = field["n_gates"]
    laps = [t for s in runs for t in s["lap_times"]]
    legs: List[List[float]] = [[] for _ in range(n)]
    for s in runs:
        for row in s["section_grid"]:
            for i, v in enumerate(row):
                if v is not None:
                    legs[i].append(v)
    sec_best = [min(v) if v else None for v in legs]

    def best(value):                                   # (the best value, the run it was flown in)
        got = [(value(s), s["stem"]) for s in runs]
        return min((g for g in got if g[0] is not None), default=(None, None))
    lap, lap_run = best(lambda s: s["best_lap"])
    row, row_run = best(lambda s: best_in_a_row(s["lap_times"], pb))
    three, _ = best(lambda s: best_in_a_row(s["lap_times"], 3))
    hole, hole_run = best(lambda s: s["holeshot"] if (s.get("holeshot") or 0) > 0 else None)
    cvs = [s["lap_cv"] for s in runs if s.get("lap_cv") is not None and s["laps"] >= 3]
    med = round(median(laps), 3) if laps else None
    tb, _ = best(lambda s: s.get("theoretical_best"))    # within one run: sections from different runs don't add up
    times = [[t for s in runs for t in named_times(s["section_grid"], x["legs"])] for x in named]
    # heat by heat: the best lap of each (the progress chart); improved = the first heat's best − the event's best
    progress = [[s.get("heat"), s["best_lap"]] for s in sorted(runs, key=lambda s: (s.get("heat") or 0, s["stem"]))
                if s["best_lap"] is not None]
    improved = round(progress[0][1] - lap, 3) if len(progress) >= 2 and progress[0][1] > lap else None
    return {"pilot": pilot, "runs": len(runs), "laps": len(laps),
            "best_lap": lap, "best_lap_run": lap_run, "best_row": row, "best_row_run": row_run, "best3": three,
            "median_lap": med, "theoretical_best": tb,
            "to_gain": round(lap - tb, 3) if lap is not None and tb is not None and lap > tb else None,
            "lap_cv": round(median(cvs), 4) if cvs else None,
            "speed_index": round(field["median_lap"] / med, 3) if med and field.get("median_lap") else None,
            "holeshot": hole, "holeshot_run": hole_run, "progress": progress, "improved": improved,
            "improved_pct": round(improved / progress[0][1], 4) if improved else None,
            "section_best": sec_best,
            "section_median": [round(median(v), 3) if v else None for v in legs],
            "named_best": [min(v) if v else None for v in times],
            "named_median": [round(median(v), 3) if v else None for v in times]}


def race_craft(heats: List[dict]) -> Dict[str, dict]:
    """Racing against others, per pilot, over the event's races (not its time trials): races flown, wins,
    the average finishing place, the average place at the holeshot, and the places gained from the holeshot
    to the finish (+1 = one place gained a race, on average). heats: DataView.heats()."""
    acc: Dict[str, dict] = {}
    for h in heats:
        kind = str(h.get("type") or "").lower()
        if not kind or "timetrial" in kind or "practi" in kind:
            continue
        placed = [p for p in h["pilots"] if p.get("position")]
        if len(placed) < 2:
            continue
        first = sorted((p for p in placed if p.get("holeshot") is not None), key=lambda p: p["holeshot"])
        hole = {p["pilot"]: i + 1 for i, p in enumerate(first)}
        for p in placed:
            a = acc.setdefault(p["pilot"], {"races": 0, "wins": 0, "places": [], "holes": [], "gained": []})
            a["races"] += 1
            a["wins"] += p["position"] == 1
            a["places"].append(p["position"])
            if p["pilot"] in hole:
                a["holes"].append(hole[p["pilot"]])
                a["gained"].append(hole[p["pilot"]] - p["position"])
    mean = lambda v: round(sum(v) / len(v), 2) if v else None
    return {name: {"races": a["races"], "wins": a["wins"], "avg_place": mean(a["places"]),
                   "avg_holeshot_place": mean(a["holes"]), "gained": mean(a["gained"])} for name, a in acc.items()}
