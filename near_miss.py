"""
near_miss.py — give the sequence decoder the crossings the pass scorer nearly kept.

The pass scorer judges every crossing on its own; a gate taken in a hard bank
(it swings out of the frame while still small) scores just below its threshold
and is dropped, so the lap shows that gate as missed. On the fril-26 / fril-27
bench about 3 in 4 missed gates were such near-misses.

label() adds every candidate with P_NEAR <= p < the scorer's threshold that
leaves through the frame edge (no image: the decoder places it by timing and
gate type alone), labels everything with gate_decoder.label_race, then drops an
added crossing that repeats a gate the lap already has from a real pass. The
added crossings it keeps are RACE passes with "recovered": true, so laps,
splits and stats count them and the UI shows them apart for a spotter to check.

The finish: a pilot who flies the last lap's start gate and lands right after it leaves a crossing the scorer
can't back up (nothing follows it), and a fast one who missed the last gates gets the start gate named one of
them. When the last lap reached its last gates and the race ends there: the last crossing, of the start gate's
type, named one of the last gates (of another type), is G1; else, the last lap having reached its last gate, the
biggest candidate within FINISH_S after it is (both "recovered", for a spotter to check).

Needs no YOLO: the candidates are rebuilt from the saved tracks + motion.
"""

import json
from pathlib import Path
from typing import List, Tuple

import cv2

import gate_decoder

P_NEAR = 0.10       # scorer probability an added crossing needs
MIN_GAP = 0.25      # seconds clear of a kept pass (the same crossing, scored twice)
REASON = "near_miss"
FINISH = "finish"
FINISH_S = 2.5      # seconds after the last gate the finish crossing can be
FINISH_AMAX = 0.3   # the box it needs to reach (area ratio): flown through, not seen far off


def scored_candidates(data: dict, runs_dir: Path, stem: str, W: int, H: int) -> List[dict]:
    """Every pass-scorer candidate of the analysis with its probability: [{t, tid, p, gtype, borders, amax}]."""
    import pass_scorer
    mf = runs_dir / f"{stem}.motion.json"
    saved = pass_scorer.load_model()
    if not mf.exists() or saved is None or "_model" not in saved:
        return []
    race = {"frames": data["frames"], "passes": data.get("live_passes") or []}
    v = pass_scorer.build_video(stem, race, json.loads(mf.read_text(encoding="utf-8")), W, H)
    v.two_stage = "decoder" in saved
    cands = pass_scorer.candidates(v)
    ratios = saved.get("type_ratios", {})
    for c in cands:
        c.feats = pass_scorer.features(v, c, ratios)
    P = saved["_model"].predict([c.feats for c in cands]) if cands else []
    return [{"t": float(c.t), "tid": int(c.tid), "p": float(p), "gtype": c.gtype,
             "borders": int(c.feats.get("borders", 0)), "amax": float(c.feats.get("amax", 0))}
            for c, p in zip(cands, P)]


def video_candidates(video: str, data: dict, tp) -> List[dict]:
    """scored_candidates of a video's analysis ([] without frames, a scorer or a readable video)."""
    import pass_scorer
    if pass_scorer.load_model() is None or not data.get("frames"):
        return []
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return scored_candidates(data, tp.runs_dir, Path(video).stem, W, H) if W and H else []


def near_misses(video: str, data: dict, tp, cands: List[dict] = None) -> List[dict]:
    """The scorer's dropped candidates worth a second look, as passes without an image."""
    import pass_scorer
    saved = pass_scorer.load_model()
    if saved is None or not data.get("frames"):
        return []
    thr = float(saved.get("threshold", 0.5))
    if cands is None:
        cands = video_candidates(video, data, tp)
    have = [p["t"] for p in data["passes"]]
    out = []
    for c in sorted(cands, key=lambda c: -c["p"]):
        if P_NEAR <= c["p"] < thr and c["borders"] >= 1 and all(abs(c["t"] - t) > MIN_GAP for t in have):
            out.append({"t": c["t"], "p_pass": round(c["p"], 4), "gate_type": c["gtype"], "track_id": c["tid"],
                        "query_embedding": None, "reason": REASON, "source": "RACE", "gate_id": -1})
            have.append(c["t"])
    return out


def _dedupe(passes: List[dict], n: int) -> int:
    """An added crossing labelled with a gate its lap already has from a real pass (or from a likelier
    added one) is not that gate: NOPASS. Returns how many were dropped."""
    added = lambda p: p.get("reason") == REASON
    dropped = 0
    for lap in gate_decoder.build_laps(passes, n):
        inside = [p for p in passes if p.get("gate_id", -1) >= 1 and lap["t0"] <= p["t"] < lap["t1"]]
        real = {p["gate_id"] for p in inside if not added(p)}
        seen = set()
        for p in sorted((p for p in inside if added(p)), key=lambda p: -(p.get("p_pass") or 0)):
            if p["gate_id"] in real or p["gate_id"] in seen:
                p["gate_id"], p["source"] = -1, "NOPASS"
                dropped += 1
            else:
                seen.add(p["gate_id"])
    return dropped


def finish(passes: List[dict], cands: List[dict], types: List[str]) -> str:
    """The last lap's start-gate crossing at the end of the race (see the module doc), in place. Returns what
    was done ("" nothing, "renamed", "added")."""
    n = len(types)
    race = sorted((p for p in passes if p.get("source") == "RACE" and p.get("gate_id", -1) >= 1),
                  key=lambda p: p["t"])
    g1 = [p["t"] for p in race if p["gate_id"] == 1]
    if not g1 or n < 4:
        return ""
    tail = [p for p in race if p["t"] > g1[-1]]       # the unfinished last lap
    if len({p["gate_id"] for p in tail}) < n // 2:
        return ""                                     # a crash early in the lap: no finish to look for
    last = tail[-1]
    if last["gate_id"] >= n - 2 and last.get("gate_type") == types[0] != types[last["gate_id"] - 1]:
        last["gate_id"], last["recovered"], last["finish"] = 1, True, True
        return "renamed"
    if last["gate_id"] != n:                          # short of the last gate: a crash more likely than a finish
        return ""
    near = [c for c in cands if last["t"] + MIN_GAP < c["t"] <= last["t"] + FINISH_S and c["amax"] >= FINISH_AMAX]
    if not near:
        return ""
    c = max(near, key=lambda c: c["amax"])
    passes.append({"t": c["t"], "p_pass": round(c["p"], 4), "gate_type": c["gtype"], "track_id": c["tid"],
                   "query_embedding": None, "reason": FINISH, "source": "RACE", "gate_id": 1, "recovered": True})
    passes.sort(key=lambda p: p["t"])
    return "added"


def label(video: str, data: dict, memory_path, tp) -> Tuple[List[dict], List[dict], dict]:
    """gate_decoder.label_race over data["passes"] plus the scorer's near-misses (data needs "frames" and
    "live_passes"), then the finish. Added crossings an earlier run left in data["passes"] are dropped first."""
    passes = [p for p in data["passes"] if p.get("reason") not in (REASON, FINISH, "lap_rescue")]
    for p in passes:
        p.pop("recovered", None)
        p.pop("finish", None)
    cands = video_candidates(video, {**data, "passes": passes}, tp) if tp is not None else []
    extra = near_misses(video, {**data, "passes": passes}, tp, cands) if tp is not None else []
    out, laps, info = gate_decoder.label_race(passes + extra, memory_path, tp)
    types = gate_decoder.load_memory(Path(memory_path))[0]
    n = len(types)
    dropped = _dedupe(out, n)
    out = [p for p in out if p.get("reason") != REASON or p["source"] == "RACE"]   # unused ones were never passes
    kept = 0
    for p in out:
        if p.get("reason") == REASON:
            p["recovered"] = True
            kept += 1
    fin = finish(out, cands, types)
    if kept or fin:
        laps = gate_decoder.build_laps(out, n)
    info["recovered"] = kept
    info["finish"] = fin
    print(f"Near-misses: {len(extra)} added, {kept} labelled as a gate, {dropped} dropped as duplicates"
          + (f"; the last lap's finish {fin}" if fin else ""))
    return out, laps, info
