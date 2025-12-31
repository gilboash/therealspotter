# pass_detector.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _norm_type(s: str) -> str:
    return (s or "").strip().lower()


def _is_flag(t: str) -> bool:
    t = _norm_type(t)
    return ("flag" in t) or ("pole" in t)


@dataclass
class TrackPassState:
    track_id: int
    ttype: str
    stage: str = "idle"          # idle -> aligned -> passed
    last_seen_time: float = 0.0

    # approach / alignment
    peak_area: float = 0.0
    last_area: float = 0.0
    last_cx: float = 0.0
    last_cy: float = 0.0

    # flag heuristics
    min_cx: float = 1e9
    max_cx: float = -1e9

    # cooldown / pass bookkeeping
    passed_time: float = 0.0


class PassDetector:
    """
    Vision-only gate pass detector, based on tracked boxes.
    Works with your Track objects (needs bbox + locked_type + score_ema).

    Call:
      pd.update(tracks, now, frame_w, frame_h)

    Then:
      pd.pop_any_passed() -> event dict (track_id, type, time, reason)
      or pd.consume_passed(track_id)
    """

    def __init__(
        self,
        min_track_score: float = 0.4,
        min_area_ratio: float = 0.030,      # gate is "close enough"
        center_tol: float = 0.18,           # normalized distance to center for "aligned"
        disappear_timeout: float = 0.25,    # seconds after alignment to count disappearance as pass
        flag_cross_min_frac: float = 0.45,  # flag center should span this fraction of width to count as wrap

        # cooldown knobs (to avoid repeated passes)
        pass_cooldown_sec: float = 1.0,     # minimum time between "pass" events overall
        type_cooldown_sec: float = 0.7,     # minimum time between passes of the same type
        track_cooldown_sec: float = 5.5,    # minimum time between passes from same track_id

        # NEW: debugging switch
        ignore_flagpoles: bool = False,     # if True, skip flagpole tracks entirely
    ):
        self.min_track_score = float(min_track_score)
        self.min_area_ratio = float(min_area_ratio)
        self.center_tol = float(center_tol)
        self.disappear_timeout = float(disappear_timeout)
        self.flag_cross_min_frac = float(flag_cross_min_frac)

        self.pass_cooldown_sec = float(pass_cooldown_sec)
        self.type_cooldown_sec = float(type_cooldown_sec)
        self.track_cooldown_sec = float(track_cooldown_sec)

        self.ignore_flagpoles = bool(ignore_flagpoles)

        self.states: Dict[int, TrackPassState] = {}
        self._just_passed: Dict[int, dict] = {}

        # cooldown history
        self._last_pass_time_global: float = -1e9
        self._last_pass_time_by_type: Dict[str, float] = {}
        self._last_pass_time_by_track: Dict[int, float] = {}

    def update(self, tracks, now: float, frame_w: int, frame_h: int):
        frame_area = float(frame_w * frame_h)
        seen_ids = set()

        # update seen tracks
        for tr in tracks:
            tid = int(tr.track_id)
            ttype = tr.locked_type or "NONE"

            # ignore junk
            if ttype == "NONE":
                continue
            if float(getattr(tr, "score_ema", 0.0)) < self.min_track_score:
                continue

            # NEW: optional filter to ignore flagpoles completely
            if self.ignore_flagpoles and _is_flag(ttype):
                continue

            seen_ids.add(tid)
            st = self.states.get(tid)
            if st is None:
                st = TrackPassState(track_id=tid, ttype=ttype)
                self.states[tid] = st

            # if type changed (rare), reset state machine
            if _norm_type(st.ttype) != _norm_type(ttype):
                st = TrackPassState(track_id=tid, ttype=ttype)
                self.states[tid] = st

            st.last_seen_time = now

            bbox = tr.bbox
            cx, cy = _center(bbox)
            a = _area(bbox)
            st.last_area = a
            st.peak_area = max(st.peak_area, a)
            st.last_cx = cx
            st.last_cy = cy

            # normalize to [0..1] for center distance
            nx = (cx / max(frame_w, 1))
            ny = (cy / max(frame_h, 1))
            dx = abs(nx - 0.5)
            dy = abs(ny - 0.5)
            center_dist = (dx * dx + dy * dy) ** 0.5

            area_ratio = a / max(frame_area, 1.0)

            # flag logic (only if not filtered above)
            if _is_flag(ttype):
                st.min_cx = min(st.min_cx, nx)
                st.max_cx = max(st.max_cx, nx)

                if area_ratio >= (self.min_area_ratio * 0.60):
                    span = st.max_cx - st.min_cx
                    if span >= self.flag_cross_min_frac:
                        self._mark_passed(st, now, reason="flag_wrap")
                continue

            # square/arch/circle: "aligned" when close + centered
            if st.stage == "idle":
                if area_ratio >= self.min_area_ratio and center_dist <= self.center_tol:
                    st.stage = "aligned"
            elif st.stage == "aligned":
                pass

        # handle tracks that disappeared: if aligned recently => PASS
        for tid, st in list(self.states.items()):
            if tid in seen_ids:
                continue

            if st.stage == "aligned":
                if (now - st.last_seen_time) <= self.disappear_timeout:
                    self._mark_passed(st, now, reason="disappear_after_align")

            if (now - st.last_seen_time) > 2.0:
                self.states.pop(tid, None)

        self._gc_cooldowns(now)

    def _cooldown_ok(self, st: TrackPassState, now: float) -> bool:
        if (now - self._last_pass_time_global) < self.pass_cooldown_sec:
            return False

        tkey = _norm_type(st.ttype)
        last_t = self._last_pass_time_by_type.get(tkey, -1e9)
        if (now - last_t) < self.type_cooldown_sec:
            return False

        last_tr = self._last_pass_time_by_track.get(st.track_id, -1e9)
        if (now - last_tr) < self.track_cooldown_sec:
            return False

        return True

    def _mark_passed(self, st: TrackPassState, now: float, reason: str):
        if st.stage == "passed":
            return
        if not self._cooldown_ok(st, now):
            return

        st.stage = "passed"
        st.passed_time = now

        self._last_pass_time_global = now
        self._last_pass_time_by_type[_norm_type(st.ttype)] = now
        self._last_pass_time_by_track[st.track_id] = now

        self._just_passed[st.track_id] = {
            "track_id": st.track_id,
            "type": st.ttype,
            "time": now,
            "reason": reason,
        }

    def _gc_cooldowns(self, now: float):
        horizon = max(5.0, self.track_cooldown_sec * 4.0)
        for tid in list(self._last_pass_time_by_track.keys()):
            if (now - self._last_pass_time_by_track[tid]) > horizon:
                self._last_pass_time_by_track.pop(tid, None)

        for tkey in list(self._last_pass_time_by_type.keys()):
            if (now - self._last_pass_time_by_type[tkey]) > horizon:
                self._last_pass_time_by_type.pop(tkey, None)

    def consume_passed(self, track_id: int) -> Optional[dict]:
        return self._just_passed.pop(int(track_id), None)

    def pop_any_passed(self) -> Optional[dict]:
        for k in list(self._just_passed.keys()):
            return self._just_passed.pop(k)
        return None
