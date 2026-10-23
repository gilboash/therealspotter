#!/usr/bin/env python3
"""
online_decoder.py — gate identity and laps while the race is flown (the Live spotter's stage 2).

gate_decoder.py labels a whole video at once, looking at every pass before and after. Live, a pass must be named
the moment it happens, from the passes before it only. OnlineDecoder runs the same dynamic program (order +
the track's learned gate-to-gate timing + CLIP appearance, a prior that the first pass is G1) one pass at a time:
each new pass adds one column, and the pass gets the gate of the best path that ends on it, or is called a false
pass when the best path ends earlier. The pilot's speed is re-estimated from the path after every pass.

Laps follow gate_decoder.build_laps: the first G1 starts lap 1, a G1 at least half a lap later ends it. When G1
itself is missed, the lap ends where the order wraps, and its time is estimated from the learned timing.

    python online_decoder.py fril-27-first-track        # simulate on the track's analysed videos, vs offline
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

import gate_decoder as gd
from dataset_paths import TrackPaths, list_tracks

SCALE_MIN, SCALE_MAX = 0.6, 1.7      # the pilot's speed vs the learned timing


class OnlineDecoder:
    def __init__(self, memory_path, timing_path: Optional[Path] = None, cfg: Optional[gd.Cfg] = None):
        self.cfg = cfg or gd.Cfg()
        types, banks = gd.load_memory(Path(memory_path))
        self.n = len(types)
        self.m = gd.fit([], self.n, types, banks, use_reviewed=False)
        self.timing = None
        if timing_path is not None and Path(timing_path).exists():
            d = json.loads(Path(timing_path).read_text(encoding="utf-8"))
            if int(d.get("n_gates", self.n)) == self.n:
                self.m.log_mu, self.m.log_sd = np.array(d["log_mu"]), np.array(d["log_sd"])
                self.timing = f"learned from {len(d.get('learned_from', []))} learn videos"
        if self.timing is None:                  # no timing known: order + appearance only
            self.cfg = gd.Cfg(**{**self.cfg.__dict__, "use_timing": False})
        self.gate_types = types
        self.reset()

    def reset(self):
        N = self.n
        cfg = self.cfg
        self.obs: List[gd.Obs] = []
        self.best: List[np.ndarray] = []         # per obs: best log-score of a path ending there, per gate (1…N)
        self.back: List[np.ndarray] = []         # per obs, per gate: (previous obs, previous gate); -1 = start
        self.scale = 1.0
        self.lf = np.log(cfg.p_false) + cfg.false_emit
        self.lr = np.log(1 - cfg.p_false)
        step = np.full(N, -np.inf)
        skips = [cfg.p_skip ** (d - 1) for d in range(2, N)]
        step[1] = np.log(max(1e-6, 1 - cfg.p_repeat - sum(skips)))
        step[0] = np.log(cfg.p_repeat)
        for d in range(2, N):
            step[d] = np.log(skips[d - 2])
        self.step = step
        self.calls: List[dict] = []              # what was announced, pass by pass
        self.lap_start: Optional[dict] = None    # the call that started the current lap
        self.laps: List[dict] = []
        self._seen = set()                       # gates called since the lap started
        self._prev_g = None

    # ── the dynamic program, one column at a time ──────────────────
    def _column(self, j: int) -> None:
        N, cfg, m = self.n, self.cfg, self.m
        E = gd.emissions(gd.VideoSeq("", [self.obs[j]]), m, cfg)[0]
        best = np.full(N + 1, -1e18)
        back = np.zeros((N + 1, 2), dtype=int)
        start = j * self.lf + self.lr
        for g in range(1, N + 1):
            best[g] = start + (cfg.start_g1 if g == 1 else 0.0) + E[g]
            back[g] = (-1, 0)
        for i in range(max(0, j - cfg.window - 1), j):
            gap = (j - i - 1) * self.lf
            dt = self.obs[j].t - self.obs[i].t
            for g0 in range(1, N + 1):
                if self.best[i][g0] <= -1e17:
                    continue
                base = self.best[i][g0] + gap + self.lr
                for g in range(1, N + 1):
                    d = (g - g0) % N
                    s = base + self.step[d] + E[g]
                    if cfg.use_timing and d > 0:
                        s += gd._timing(m, g0, d, dt, self.scale)
                    if s > best[g]:
                        best[g], back[g] = s, (i, g0)
        self.best.append(best)
        self.back.append(back)

    def _path(self, j: int, g: int) -> List[int]:
        labels = [0] * (j + 1)
        while j >= 0:
            labels[j] = g
            j, g = self.back[j][g]
        return labels

    def _update_scale(self, labels: List[int]):
        if not (self.cfg.use_timing and self.timing):
            return
        r = []
        real = [(o.t, g) for o, g in zip(self.obs, labels) if g > 0]
        for (t0, g0), (t1, g1) in zip(real, real[1:]):
            if (g1 - g0) % self.n == 1 and t1 > t0:
                r.append(np.log(t1 - t0) - self.m.log_mu[g0])
        if len(r) >= 3:
            self.scale = float(np.clip(np.exp(np.median(r)), SCALE_MIN, SCALE_MAX))

    def expected_leg(self, g0: int, d: int = 1) -> float:
        """Learned seconds from gate g0 forward d gates, at this pilot's speed."""
        return float(sum(np.exp(self.m.log_mu[(g0 - 1 + k) % self.n + 1]) for k in range(d)) * self.scale)

    # ── a pass ─────────────────────────────────────────────────
    def add(self, t: float, emb=None, gate_type: str = "") -> List[dict]:
        """A new pass at t (seconds, feed time). Returns the events it causes: one "gate" (gate_id 0 = judged
        not a real pass), then a "lap" when it closes a lap, and "start" when a lap starts (at = the time of the
        pass that called it; t is earlier when G1 was missed and the start is estimated)."""
        self.obs.append(gd.Obs(float(t), gd._norm(emb) if emb is not None else None, str(gate_type)))
        j = len(self.obs) - 1
        self._column(j)
        # the best path so far: it may end on this pass, or on an earlier one with the rest false
        cands = [(self.best[i][g] + (j - i) * self.lf, i, g) for i in range(max(0, j - self.cfg.window), j + 1)
                 for g in range(1, self.n + 1)]
        score, i, g = max(cands)
        labels = self._path(i, g) + [0] * (j - i)
        self._update_scale(labels)
        gid = labels[j]
        alt = sorted(self.best[j][1:], reverse=True)
        conf = float(1 / (1 + np.exp(-(alt[0] - alt[1])))) if gid and len(alt) > 1 else None
        call = {"type": "gate", "t": round(float(t), 3), "gate_id": int(gid),
                "conf": round(conf, 3) if conf is not None else None, "scale": round(self.scale, 3)}
        self.calls.append(call)
        out = [call]
        if gid:
            out += self._laps(call)
            self._prev_g = gid
        return out

    def _wrapped(self, g: int) -> bool:
        """G1, or the order went back across the lap's end (G13 → G2), not a label stepping back (G6 → G5)."""
        return g == 1 or (self._prev_g is not None and self._prev_g - g >= self.n // 2)

    def _laps(self, call: dict) -> List[dict]:
        g = call["gate_id"]
        if self.lap_start is None:
            return [self._open(call, g)] if self._wrapped(g) else []
        if g == 1 and self._seen <= {1}:             # G1 again right away: the later crossing starts the lap
            self.lap_start = {"t": call["t"], "gate_id": 1, "estimated": False}
            return []
        if self._wrapped(g) and len(self._seen | {1}) >= self.n // 2:
            t0 = self.lap_start["t"]
            t1, est = (call["t"], False) if g == 1 else (call["t"] - self.expected_leg(1, g - 1), True)
            lap = {"type": "lap", "lap": len(self.laps) + 1, "t0": round(t0, 3), "t1": round(t1, 3),
                   "dt": round(t1 - t0, 3), "estimated": bool(est or self.lap_start["estimated"]),
                   "gates": len(self._seen | {1}), "at": call["t"]}
            self.laps.append(lap)
            return [lap, self._open(call, g, t1, est)]
        self._seen.add(g)
        return []

    def _open(self, call: dict, g: int, t0: Optional[float] = None, est: Optional[bool] = None) -> dict:
        if t0 is None:
            t0, est = (call["t"], False) if g == 1 else (call["t"] - self.expected_leg(1, g - 1), True)
        self.lap_start = {"t": t0, "gate_id": g, "estimated": est}
        self._seen = {g}
        return {"type": "start", "t": round(t0, 3), "lap": len(self.laps) + 1, "estimated": bool(est), "at": call["t"]}

    def final(self) -> List[int]:
        """The labels the offline decoder gives these same passes (every pass seen), for the summary."""
        v = gd.VideoSeq("", list(self.obs))
        return gd.decode_video(v, self.m, self.cfg) if self.obs else []


def for_track(tp: TrackPaths) -> OnlineDecoder:
    return OnlineDecoder(tp.gate_memory, tp.learned_timing)


# ──────────────────────────────────────────────────────────────
# Simulation on analysed videos: the live detector's passes, one at a time, vs the offline result
# ──────────────────────────────────────────────────────────────

def simulate(tp: TrackPaths, window: float = 0.35, verbose: bool = False):
    dec = for_track(tp)
    print(f"{tp.name}: {dec.n} gates, timing {dec.timing}\n")
    tot = {"g_ok": 0, "g_n": 0, "lap_ref": 0, "lap_found": 0, "lap_extra": 0, "err": [], "est": 0, "start_ok": 0,
           "videos": 0}
    for f in sorted(tp.runs_dir.glob("*.race_data.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        live = sorted(d.get("live_passes") or [], key=lambda p: p["t"])
        ref = [p for p in d.get("passes", []) if p.get("source") == "RACE" and p.get("gate_id", -1) >= 1]
        if not live or not ref:
            continue
        dec.reset()
        laps, starts = [], []
        for p in live:
            for e in dec.add(p["t"], p.get("query_embedding"), p.get("gate_type", "")):
                if e["type"] == "lap":
                    laps.append(e)
                elif e["type"] == "start":
                    starts.append(e)
        # gate labels vs the offline labels of the same crossings
        ok = n = 0
        for c in dec.calls:
            r = min(ref, key=lambda p: abs(p["t"] - c["t"]))
            if abs(r["t"] - c["t"]) <= window:
                n += 1
                ok += int(r["gate_id"] == c["gate_id"])
        # laps vs offline laps (by end time)
        rlaps = d.get("laps", [])
        used, errs = set(), []
        for rl in rlaps:
            k = min((k for k in range(len(laps)) if k not in used and abs(laps[k]["t1"] - rl["t1"]) <= 1.0),
                    key=lambda k: abs(laps[k]["t1"] - rl["t1"]), default=None)
            if k is not None:
                used.add(k)
                errs.append(laps[k]["dt"] - rl["dt"])
        rstart = rlaps[0]["t0"] if rlaps else None
        start_ok = bool(starts and rstart is not None and abs(starts[0]["t"] - rstart) <= 1.0)
        tot["g_ok"] += ok; tot["g_n"] += n; tot["lap_ref"] += len(rlaps); tot["lap_found"] += len(errs)
        tot["lap_extra"] += len(laps) - len(errs); tot["err"] += errs; tot["start_ok"] += start_ok
        tot["est"] += sum(1 for l in laps if l["estimated"]); tot["videos"] += 1
        stem = f.name[: -len(".race_data.json")]
        print(f"  {stem[:38]:38s} gates {ok:3d}/{n:3d}  laps {len(errs)}/{len(rlaps)} (+{len(laps) - len(errs)} extra)"
              f"  start {'ok' if start_ok else '--'}  lap err " +
              (f"median {1000 * float(np.median(np.abs(errs))):.0f} ms, max {1000 * float(np.max(np.abs(errs))):.0f} ms"
               if errs else "-") + f"  speed {dec.scale:.2f}")
        if verbose:
            for l in laps:
                print(f"      lap {l['lap']}: {l['dt']:.2f}s{' ~' if l['estimated'] else ''}")
            print("      offline:", ", ".join(f"{l['dt']:.2f}" for l in rlaps))
    e = np.abs(tot["err"]) if tot["err"] else np.zeros(1)
    print(f"\nTotal over {tot['videos']} videos: gate id {tot['g_ok']}/{tot['g_n']} = "
          f"{100 * tot['g_ok'] / max(1, tot['g_n']):.1f}% (same as offline); laps {tot['lap_found']}/{tot['lap_ref']} found, "
          f"{tot['lap_extra']} extra, {tot['est']} with an estimated G1; lap time error median {1000 * np.median(e):.0f} ms, "
          f"95% {1000 * np.percentile(e, 95):.0f} ms; first lap started right in {tot['start_ok']}/{tot['videos']}")
    return tot


def main():
    ap = argparse.ArgumentParser(description="Simulate the online gate decoder on a track's analysed videos")
    ap.add_argument("track")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    tp = next(t for t in list_tracks() if t.dir.name == args.track)
    simulate(tp, verbose=args.verbose)


if __name__ == "__main__":
    main()
