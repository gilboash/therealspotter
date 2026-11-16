"""
Marshal assist: a pilot's laps from the timing system (LapRF) against the
laps seen in the video.

A lap boundary in the video is where the pilot crosses the start/finish
gate. It's found from the gate sequence, not only from G1 itself (the
detector misses some passes): a G1 pass gives the exact moment; G13 → G2
with G1 not seen means the line was crossed somewhere between those two
passes. Each boundary is a time window [a, b].

Only disagreements are reported, each with a confidence:

  missed    the video shows the pilot cross the start/finish where the
            timing system has no detection: G1 seen, a whole lap (most gates)
            in a believable time, and the pilot flies on afterwards → a lap
            to add
  false     a timing detection with an impossible lap time and no
            start/finish crossing near it in the video → a lap to remove
  unsure    anything else that doesn't match: the video can't tell for
            sure (a missed or mislabelled gate, a crash at the line, a bad
            picture, a DJI feed…) — for a person to look at

Only "missed" and "false" are high confidence, and those are the only ones
Auto mode writes.

The timing system is the reference (FRIL Race 2: LapRF was right far more
often than the video): a timing lap the video doesn't confirm, or a video
crossing the timing doesn't have, is only reported when the lap time itself
is odd (shorter than ODD_SHORT or longer than ODD_LONG × the usual lap). A
crossing FPVTrackside wouldn't count anyway (under its minimum lap time…) is
not reported at all. On 2027Race1 every looser rule (e.g. "the video shows
the pilot mid-lap") produced wrong corrections, so the rules stay strict.
A lap is never added where FPVTrackside wouldn't count one (see rules in
check_pilot): Training 1 2027 had 31 wrong additions, all of them a second
copy of a crossing the timing already had, or a lap after the pilot's race.

Times: video passes are converted to the clock with the heat's frame times,
plus an offset (the timing antenna isn't exactly at the visual gate),
measured per heat (heat_offset). The antenna can fire anywhere from the
start/finish gate to the gate after it (Training 1 2027: up to 1.6 s after
G1), so a timing crossing from EARLY_S before to LATE_S after the video's
crossing is the same crossing.
"""

from statistics import median
from typing import List, Optional

TOL_S = 1.0           # a marshal's mark this close to a timing crossing is that crossing
EARLY_S = 0.7         # a timing crossing this much before the video's start/finish crossing…
LATE_S = 2.5          # …or this much after it is the same crossing
PAIR_S = 2.5          # window used to measure the offset
MAX_GAP_S = 6.0       # an inferred boundary (G1 not seen) wider than this isn't trusted
MIN_GATES = 0.6       # a lap in the video needs this share of the gates
SHORT = 0.5           # a lap shorter than this × the usual lap is impossible
EXACT_G1_S = 2.0      # G1 passes this close are one exact crossing
SAME_G1_S = 5.0       # G1 passes one after another (no other gate between) this close are one crossing
ODD_SHORT = 0.75      # a timing lap this much shorter…
ODD_LONG = 1.6        # …or longer than the usual lap is odd: only then is a disagreement worth a look
HOLESHOT_S = 0.5      # the first gate after the start: passes from this long before the start on
MAX_OFFSET_S = 2.5    # a heat's timing − video offset beyond this isn't believed (the event's is used)


def g1_passes(race: dict) -> List[dict]:
    return sorted((p for p in race.get("passes", []) if p.get("gate_id") == 1), key=lambda p: p["t"])


def pairs_offset(det_times: List[float], video_clock: List[float]) -> List[float]:
    """Detection − video differences of the closest pairs (for the offset)."""
    out = []
    for d in det_times:
        if video_clock:
            v = min(video_clock, key=lambda x: abs(x - d))
            if abs(d - v) <= PAIR_S:
                out.append(d - v)
    return out


def boundaries(passes: List[dict], n_gates: int) -> List[dict]:
    """Start/finish crossings in the video: {"a", "b" (video s), "exact",
    "sure", "gates": distinct gates seen since the previous crossing}.
    G1 passes one after another a few seconds apart are one crossing (the
    gate before G1 taken for G1, or G1 seen twice); it's exact when they're
    close. A G1 is "sure" when the pass before it is the track's last gate
    (or there's none): a G1 right after a middle gate may be the last gate
    mislabelled."""
    ps = sorted((p for p in passes if p.get("gate_id", -1) >= 1), key=lambda p: p["t"])
    top = max((p["gate_id"] for p in ps), default=n_gates)
    out, seen = [], set()
    prev = None
    for p in ps:
        g = p["gate_id"]
        if g == 1:
            if out and out[-1]["g1"] and prev is not None and prev["gate_id"] == 1 \
                    and p["t"] - out[-1]["a"] < SAME_G1_S:
                b = out[-1]
                b["b"] = p["t"]
                b["exact"] = b["exact"] and p["t"] - b["a"] < EXACT_G1_S
                prev = p
                continue                                  # the same crossing seen twice
            out.append({"a": p["t"], "b": p["t"], "exact": True, "g1": True, "gates": len(seen),
                        "sure": prev is None or prev["gate_id"] == top})
            seen = set()
        elif prev is not None and prev["gate_id"] != 1 and g < prev["gate_id"]:
            # the sequence wrapped (… G13, G14 → G2 …): the line was crossed in between
            out.append({"a": prev["t"], "b": p["t"], "exact": False, "g1": False, "gates": len(seen),
                        "sure": False})
            seen = set()
        seen.add(g)
        prev = p
    return out


def _mid_lap(passes: List[dict], t: float, n_gates: int, win: float = 2.0) -> bool:
    """The pilot is clearly in the middle of the lap at video time t: passes
    of middle gates (not near the start/finish) just before and just after."""
    lo, hi = 3, n_gates - 1
    before = [p for p in passes if t - win <= p["t"] < t and lo <= p.get("gate_id", -1) <= hi]
    after = [p for p in passes if t < p["t"] <= t + win and lo <= p.get("gate_id", -1) <= hi]
    return bool(before and after)


NO_VIDEO_SHARE = 0.8   # no picture for this share of the race → the run can't be checked ("no video");
                       # a run with less is checked where it has a picture


def _share(intervals, a: float, b: float) -> float:
    if b <= a:
        return 0.0
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in intervals) / (b - a)


def check_pilot(race: dict, dets: List[dict], to_clock, n_gates: int, offset: float,
                start: float, end: float, lap_ref: Optional[float], low_quality: bool = False,
                gray: Optional[List[List[float]]] = None, race_video: Optional[tuple] = None,
                rejected: Optional[List[float]] = None, rules: Optional[dict] = None,
                trust_timing: bool = True) -> dict:
    """race: the pilot's labelled race_data; dets: the timing system's valid
    detections for this channel ({"t", "holeshot", "id"}); to_clock: video
    seconds → clock seconds. gray: video intervals with no picture (a flat
    gray feed); race_video: the race's start and end in video seconds.
    rejected: clock times of this channel's crossings FPVTrackside doesn't
    count (a crossing there is taken: no lap is added on it). rules: what
    FPVTrackside counts as a lap — {"min_lap", "race_length" (s), "race_type"
    (Race, TimeTrial…), "target_laps", "holeshot"}; a lap it wouldn't count is
    never added: shorter than min_lap (either side of it), in a Race after the
    pilot's target laps, in a time trial started after the race length.
    trust_timing: the timing system is the reference — a disagreement is only
    reported when the lap time is odd (see the module doc).
    Returns laps from both sides, the findings, and `confirmed`: the timing
    crossings the video showed (a crossing neither confirmed nor in a finding
    wasn't checked — e.g. while there was no picture). With no picture for most
    of the race the run is "no video" and nothing is reported; otherwise
    nothing is reported while the picture is missing."""
    gray = gray or []
    if race_video and _share(gray, *race_video) >= NO_VIDEO_SHARE:
        return {"timing_laps": max(0, len(dets) - 1), "video_laps": 0, "matched": 0, "findings": [],
                "no_video": True, "confirmed": []}
    passes = [p for p in race.get("passes", []) if p.get("gate_id", -1) >= 1]
    lo, hi = start - 2.0, end + 30.0
    bnd = []
    for b in boundaries(passes, n_gates):
        ca, cb = float(to_clock(b["a"])) + offset, float(to_clock(b["b"])) + offset
        if lo <= cb and ca <= hi and (b["exact"] or cb - ca <= MAX_GAP_S):
            bnd.append(dict(b, ca=ca, cb=cb, video_t=b["b"] if b["exact"] else (b["a"] + b["b"]) / 2))
    det = [d for d in dets if lo <= d["t"] <= hi]

    rules = rules or {}
    rejected = [t for t in (rejected or []) if lo <= t <= hi]

    def near(t, b):                              # the timing crossing at t is the video's crossing b
        return b["ca"] - EARLY_S <= t <= b["cb"] + LATE_S

    # match detections to boundary windows (closest first, one to one)
    def dist(d, b):
        return 0.0 if b["ca"] <= d["t"] <= b["cb"] else min(abs(d["t"] - b["ca"]), abs(d["t"] - b["cb"]))
    cand = sorted((dist(d, b), i, j) for i, d in enumerate(det) for j, b in enumerate(bnd) if near(d["t"], b))
    md, mb = {}, {}
    for _, i, j in cand:
        if i not in md and j not in mb:
            md[i], mb[j] = j, i

    det_times = [d["t"] for d in det]
    laps_det = [b - a for a, b in zip(det_times, det_times[1:])]
    # the usual lap: the pilot's own (a fast pilot laps well under the track's learned lap — FRIL Race 2:
    # 15 s against 30 s), but never under half the track's (a channel LapRF triggers every few seconds)
    own = sorted(laps_det)[(len(laps_det) - 1) // 2] if laps_det else None   # the lower median: a lap
                                                 # LapRF cut in two or joined (a missed crossing) moves it least
    ref = (max(own, 0.5 * lap_ref) if own and lap_ref else own or lap_ref)
    trust_timing = trust_timing and len(det) >= 2      # LapRF has (almost) nothing: nothing to trust
    findings = []
    clock0 = float(to_clock(0.0))

    def video_time(clock_t):                     # video time of a clock time (for jumping there)
        return clock_t - offset - clock0

    def odd(lap):                                # a lap time worth a look
        return lap is not None and bool(ref) and (lap < ODD_SHORT * ref or lap > ODD_LONG * ref)

    # timing detections the video doesn't confirm
    for i, d in enumerate(det):
        if i in md:
            continue
        prev = det[i - 1]["t"] if i > 0 else None
        lap = d["t"] - prev if prev is not None else None
        vt = video_time(d["t"])
        f = {"kind": "false", "t": d["t"], "detection_id": d["id"], "lap_time": lap,
             "holeshot": d["holeshot"], "video_t": vt}
        near_crossing = any(b["ca"] - 3.0 <= d["t"] <= b["cb"] + 3.0 for b in bnd)
        if lap is not None and ref and lap < SHORT * ref and not near_crossing:
            f.update(confidence="high", reason=f"lap of {lap:.1f}s is impossible (usual {ref:.1f}s) "
                                                "and the video shows no start/finish crossing")
        elif trust_timing and not odd(lap):
            continue                             # the timing is trusted: an ordinary lap needs no look
        elif _mid_lap(passes, vt, n_gates):
            f.update(kind="unsure", confidence="low",
                     reason="the video seems to show the pilot mid-lap here — check it")
        else:
            f.update(kind="unsure", confidence="low",
                     reason="timing lap not confirmed by the video — check it")
        findings.append(f)

    # video lap boundaries the timing system doesn't have
    for j, b in enumerate(bnd):
        if j in mb:
            continue
        prev_t = max([x["t"] for x in det if x["t"] < b["ca"]] +
                     [x["cb"] for x in bnd[:j]], default=None)
        lap = b["cb"] - prev_t if prev_t is not None else None
        holeshot = not det or b["cb"] < det[0]["t"] - TOL_S
        t_add = b["ca"] if b["exact"] else (b["ca"] + b["cb"]) / 2      # a repeated G1: its first pass
        f = {"kind": "missed", "t": t_add, "video_t": b["video_t"],
             "lap_time": lap, "holeshot": holeshot, "gates_seen": b["gates"], "exact": b["exact"]}
        good = (b["exact"] and b["sure"] and j > 0 and b["gates"] >= MIN_GATES * n_gates
                and ref and lap is not None and 0.75 * ref <= lap <= 1.6 * ref)
        flies_on = len({p["gate_id"] for p in passes
                        if b["b"] < p["t"] <= b["b"] + 10.0 and p["gate_id"] != 1}) >= 2
        refused = _not_a_lap(t_add, b, det, rejected, bnd[j + 1:], rules, start, near)
        before = [x["t"] for x in det if x["t"] < b["ca"]]
        after = [x["t"] for x in det if x["t"] > b["cb"]]
        in_long_lap = bool(before and after and ref and after[0] - before[-1] > ODD_LONG * ref)
        if trust_timing and refused and "minimum lap" in refused:
            continue                             # a second copy of a crossing: FPVTrackside wouldn't count it
        last = bool(before) and not after         # after LapRF's last crossing: a missed last lap?
        if trust_timing and not (good or in_long_lap or last or refused):
            continue                             # inside an ordinary timing lap: the timing is trusted
        if refused:
            f.update(kind="unsure", confidence="low",
                     reason=f"a start/finish crossing the timing doesn't have, but {refused} — check it")
        elif good and flies_on and not low_quality:
            f.update(confidence="high", reason=f"lap completed in the video ({b['gates']}/{n_gates} gates, "
                                                f"{lap:.1f}s) and the pilot flies on, but not in the timing")
        elif good:
            f.update(kind="unsure", confidence="low", reason="a lap completed at the end of the flight that the "
                     "timing doesn't have (a crash at the line?) — check it")
        else:
            f.update(kind="unsure", confidence="low", reason="a start/finish crossing in the video that the "
                     "timing doesn't have — check it")
        findings.append(f)

    blind = [f for f in findings if f.get("video_t") is not None
             and any(a - 1.0 <= f["video_t"] <= b + 1.0 for a, b in gray)]
    findings = [f for f in findings if f not in blind]
    findings.sort(key=lambda f: f["t"])
    return {"timing_laps": max(0, len(det) - 1), "video_laps": max(0, len(bnd) - 1),
            "matched": len(md), "findings": findings, "no_video_findings": len(blind),
            "confirmed": [det[i]["id"] for i in md]}        # the timing crossings the video showed


def _not_a_lap(t: float, b: dict, det: List[dict], rejected: List[float], later: List[dict], rules: dict,
               start: float, near) -> Optional[str]:
    """Why FPVTrackside wouldn't count a lap added at t (the video's crossing
    b), or None."""
    if any(near(r, b) for r in rejected):
        return "FPVTrackside discounted a timing crossing there"
    min_lap = rules.get("min_lap")
    before = [d["t"] for d in det if d["t"] < t]
    after = [d["t"] for d in det if d["t"] > t] + [x["ca"] for x in later]
    if min_lap and ((before and t - before[-1] < min_lap) or (after and min(after) - t < min_lap)):
        return f"a lap there would be shorter than the minimum lap time ({min_lap:.0f} s)"
    kind, target = rules.get("race_type"), rules.get("target_laps")
    laps_done = len(before) - (1 if rules.get("holeshot", True) and before else 0)
    if kind == "Race" and target and laps_done >= target:
        return f"the pilot had already finished the race ({target} laps)"
    length = rules.get("race_length")
    if kind == "TimeTrial" and length and before and before[-1] - start > length:
        return "the lap starts after the race time ran out"
    return None


def first_pass(passes: List[dict], start_video: float) -> Optional[dict]:
    """The pilot's first gate pass after the race start (video seconds): the
    holeshot, i.e. the start/finish gate — the pilots line up behind it (FRIL
    Race 2: the first gate after the start was G1 in 102 of 117 runs, and
    LapRF's holeshot was on it within 0.05 s in half of them)."""
    ps = sorted((p for p in passes if p["t"] >= start_video - HOLESHOT_S), key=lambda p: p["t"])
    return ps[0] if ps else None


def holeshot_offset(diffs: List[float]) -> Optional[float]:
    """Timing − video from the heat's holeshots (each pilot's first timing
    crossing against their first gate pass after the start): the median when
    at least two pilots agree within 0.5 s, else None. Unlike heat_offset it
    doesn't need the gates' numbering to be right."""
    if len(diffs) < 2:
        return None
    m = median(diffs)
    close = [d for d in diffs if abs(d - m) <= 0.5]
    return round(median(close), 3) if len(close) >= 2 else None


def numbering_shift(first_gates: List[int]) -> Optional[int]:
    """The gate the track's numbering calls the start/finish, when it isn't
    G1: the first gate after the start, as most pilots' runs label it (at
    least 3 runs, more than half agreeing on the same gate other than 1).
    None when the numbering looks right (or can't be told)."""
    gs = [g for g in first_gates if g and g >= 1]
    if len(gs) < 3:
        return None
    top = max(set(gs), key=gs.count)
    return top if top != 1 and gs.count(top) * 2 > len(gs) else None


def heat_offset(diffs: List[float]) -> Optional[float]:
    """Timing − video at the start/finish, from the heat's paired crossings:
    the lower quartile — the antenna fires at the gate or (often) after it, so
    the early pairs are the gate's own time."""
    if len(diffs) < 3:
        return None
    xs = sorted(diffs)
    return round(xs[(len(xs) - 1) // 4], 3)
