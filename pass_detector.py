# pass_detector.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any


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
    stage: str = "idle"  # idle -> aligned -> passed
    last_seen_time: float = 0.0

    # approach / alignment (pixel-space)
    peak_area: float = 0.0
    last_area: float = 0.0
    last_cx: float = 0.0
    last_cy: float = 0.0

    # flag heuristics (normalized x span)
    min_cx: float = 1e9
    max_cx: float = -1e9

    # cooldown / pass bookkeeping
    passed_time: float = 0.0

    # ----------------------------
    # Debug values (per update)
    # ----------------------------
    last_nx: float = 0.0
    last_ny: float = 0.0
    last_center_dist: float = 0.0
    last_area_ratio: float = 0.0
    last_flag_span: float = 0.0

    last_score_ema: float = 0.0
    last_reason_attempted: str = ""
    last_pass_blocked_by: str = ""  # "", "global", "type", "track", "already_passed"
    last_updated_time: float = 0.0

    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # ----------------------------
    # NEW: growth-based stats
    # ----------------------------
    prev_area_ratio: float = 0.0
    area_vel: float = 0.0          # (area_ratio delta) / sec
    area_vel_ema: float = 0.0      # smoothed velocity
    aligned_area_ratio: float = 0.0
    aligned_time: float = 0.0


class PassDetector:
    """
    Vision-only gate pass detector, based on tracked boxes.
    Works with your Track objects (needs bbox + locked_type + score_ema).

    Call:
      pd.update(tracks, now, frame_w, frame_h)

    Then:
      pd.pop_any_passed() -> event dict (track_id, type, time, reason)
      pd.get_debug(now) -> dict[track_id] of debug stats for visualization
    """

    def __init__(
        self,
        min_track_score: float = 0.22,
        min_area_ratio: float = 0.030,     # gate is "close enough" (3% of frame is typical)
        center_tol: float = 0.18,          # normalized distance to center for "aligned"
        disappear_timeout: float = 0.25,   # seconds after alignment to count disappearance as pass
        flag_cross_min_frac: float = 0.45, # flag center should span this fraction of width to count as wrap

        # cooldown knobs
        pass_cooldown_sec: float = 1.0,
        type_cooldown_sec: float = 0.7,
        track_cooldown_sec: float = 5.5,

        # behavior switches
        ignore_flagpoles: bool = False,

        # ----------------------------
        # NEW: growth-based alignment knobs
        # ----------------------------
        area_vel_ema_alpha: float = 0.35,     # smoothing for area velocity
        min_area_vel_ema: float = 0.015,      # min growth rate (area_ratio/sec) to arm alignment
        aligned_shrink_reset_frac: float = 0.18,  # if area drops by this fraction from aligned_area => reset to idle
        aligned_max_age_sec: float = 1.2,     # if aligned too long without pass, reset to idle (optional safety)
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

        self.area_vel_ema_alpha = float(area_vel_ema_alpha)
        self.min_area_vel_ema = float(min_area_vel_ema)
        self.aligned_shrink_reset_frac = float(aligned_shrink_reset_frac)
        self.aligned_max_age_sec = float(aligned_max_age_sec)

        self.states: Dict[int, TrackPassState] = {}
        self._just_passed: Dict[int, dict] = {}

        # cooldown history
        self._last_pass_time_global: float = -1e9
        self._last_pass_time_by_type: Dict[str, float] = {}
        self._last_pass_time_by_track: Dict[int, float] = {}

    # ----------------------------
    # Public debug API
    # ----------------------------
    def get_debug(self, now: float) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for tid, st in self.states.items():
            tkey = _norm_type(st.ttype)

            rem_global = max(0.0, self.pass_cooldown_sec - (now - self._last_pass_time_global))
            rem_type = max(0.0, self.type_cooldown_sec - (now - self._last_pass_time_by_type.get(tkey, -1e9)))
            rem_track = max(0.0, self.track_cooldown_sec - (now - self._last_pass_time_by_track.get(tid, -1e9)))

            out[tid] = {
                "track_id": st.track_id,
                "type": st.ttype,
                "stage": st.stage,
                "last_seen_age": float(now - st.last_seen_time),

                "score_ema": float(st.last_score_ema),

                "area_ratio": float(st.last_area_ratio),
                "center_dist": float(st.last_center_dist),
                "nx": float(st.last_nx),
                "ny": float(st.last_ny),

                "flag_span": float(st.last_flag_span),

                # growth-based debug
                "area_vel": float(st.area_vel),
                "area_vel_ema": float(st.area_vel_ema),
                "min_area_vel_ema": float(self.min_area_vel_ema),
                "aligned_area_ratio": float(st.aligned_area_ratio),
                "aligned_age": float(now - st.aligned_time) if st.aligned_time > 0 else 0.0,

                # thresholds
                "min_area_ratio": float(self.min_area_ratio),
                "center_tol": float(self.center_tol),
                "disappear_timeout": float(self.disappear_timeout),
                "flag_cross_min_frac": float(self.flag_cross_min_frac),
                "aligned_shrink_reset_frac": float(self.aligned_shrink_reset_frac),
                "aligned_max_age_sec": float(self.aligned_max_age_sec),

                "cooldown_remaining_global": float(rem_global),
                "cooldown_remaining_type": float(rem_type),
                "cooldown_remaining_track": float(rem_track),

                "last_reason_attempted": st.last_reason_attempted,
                "last_pass_blocked_by": st.last_pass_blocked_by,

                "bbox": st.last_bbox,
            }
        return out

    # ----------------------------
    # Main logic
    # ----------------------------
    def update(self, tracks, now: float, frame_w: int, frame_h: int):
        frame_area = float(frame_w * frame_h)
        seen_ids = set()

        # update seen tracks
        for tr in tracks:
            tid = int(tr.track_id)
            ttype = tr.locked_type or "NONE"
            score_ema = float(getattr(tr, "score_ema", 0.0))

            # ignore junk
            if ttype == "NONE":
                continue
            if score_ema < self.min_track_score:
                continue
            if self.ignore_flagpoles and _is_flag(ttype):
                continue

            seen_ids.add(tid)

            st = self.states.get(tid)
            if st is None:
                st = TrackPassState(track_id=tid, ttype=ttype)
                st.last_updated_time = now
                self.states[tid] = st

            # if type changed (rare), reset state machine
            if _norm_type(st.ttype) != _norm_type(ttype):
                st = TrackPassState(track_id=tid, ttype=ttype)
                st.last_updated_time = now
                self.states[tid] = st

            # time delta for velocity
            dt = max(1e-6, float(now - st.last_updated_time))
            st.last_updated_time = now

            st.last_seen_time = now
            st.last_score_ema = score_ema

            bbox = tr.bbox
            st.last_bbox = bbox

            cx, cy = _center(bbox)
            a = _area(bbox)
            st.last_area = a
            st.peak_area = max(st.peak_area, a)
            st.last_cx = cx
            st.last_cy = cy

            # normalize to [0..1]
            nx = (cx / max(frame_w, 1))
            ny = (cy / max(frame_h, 1))
            dx = abs(nx - 0.5)
            dy = abs(ny - 0.5)
            center_dist = (dx * dx + dy * dy) ** 0.5
            area_ratio = a / max(frame_area, 1.0)

            st.last_nx = nx
            st.last_ny = ny
            st.last_center_dist = center_dist
            st.last_area_ratio = area_ratio

            st.last_reason_attempted = ""
            st.last_pass_blocked_by = ""

            # --------------------------------
            # Growth stats update
            # --------------------------------
            prev = float(st.prev_area_ratio)
            vel = (area_ratio - prev) / dt
            st.area_vel = vel
            st.area_vel_ema = (1.0 - self.area_vel_ema_alpha) * st.area_vel_ema + self.area_vel_ema_alpha * vel
            st.prev_area_ratio = area_ratio

            # --------------------------------
            # flag logic
            # --------------------------------
            if _is_flag(ttype):
                st.min_cx = min(st.min_cx, nx)
                st.max_cx = max(st.max_cx, nx)
                span = st.max_cx - st.min_cx
                st.last_flag_span = span

                # require being close-ish at some point + wide span
                if area_ratio >= (self.min_area_ratio * 0.60):
                    if span >= self.flag_cross_min_frac:
                        st.last_reason_attempted = "flag_wrap"
                        self._mark_passed(st, now, reason="flag_wrap")
                continue

            st.last_flag_span = 0.0

            # --------------------------------
            # Growth-based alignment logic
            # --------------------------------
            growth_ok = (st.area_vel_ema >= self.min_area_vel_ema)

            if st.stage == "idle":
                # Only arm alignment if we're approaching (growing) + centered + close enough
                if area_ratio >= self.min_area_ratio and center_dist <= self.center_tol and growth_ok:
                    st.stage = "aligned"
                    st.aligned_area_ratio = area_ratio
                    st.aligned_time = now

            elif st.stage == "aligned":
                # If it's shrinking significantly relative to when we aligned, we're likely looking at the gate behind us.
                if st.aligned_area_ratio > 0:
                    drop_frac = (st.aligned_area_ratio - area_ratio) / max(st.aligned_area_ratio, 1e-9)
                    if drop_frac >= self.aligned_shrink_reset_frac:
                        # disarm: moving away, don't let disappearance count as pass
                        st.stage = "idle"
                        st.aligned_area_ratio = 0.0
                        st.aligned_time = 0.0

                # Optional: if aligned too long, reset (prevents stale aligned tracks)
                if st.stage == "aligned" and self.aligned_max_age_sec > 0:
                    if st.aligned_time > 0 and (now - st.aligned_time) > self.aligned_max_age_sec:
                        st.stage = "idle"
                        st.aligned_area_ratio = 0.0
                        st.aligned_time = 0.0

        # handle disappeared tracks: if aligned recently => PASS
        for tid, st in list(self.states.items()):
            if tid in seen_ids:
                continue

            if st.stage == "aligned":
                # disappeared quickly after alignment => likely passed through
                if (now - st.last_seen_time) <= self.disappear_timeout:
                    st.last_reason_attempted = "disappear_after_align"
                    self._mark_passed(st, now, reason="disappear_after_align")

            # cleanup old states
            if (now - st.last_seen_time) > 2.0:
                self.states.pop(tid, None)

        self._gc_cooldowns(now)

    # ----------------------------
    # Cooldown helpers
    # ----------------------------
    def _cooldown_ok(self, st: TrackPassState, now: float) -> Tuple[bool, str]:
        if st.stage == "passed":
            return (False, "already_passed")

        if (now - self._last_pass_time_global) < self.pass_cooldown_sec:
            return (False, "global")

        tkey = _norm_type(st.ttype)
        last_t = self._last_pass_time_by_type.get(tkey, -1e9)
        if (now - last_t) < self.type_cooldown_sec:
            return (False, "type")

        last_tr = self._last_pass_time_by_track.get(st.track_id, -1e9)
        if (now - last_tr) < self.track_cooldown_sec:
            return (False, "track")

        return (True, "")

    def _mark_passed(self, st: TrackPassState, now: float, reason: str):
        ok, blocked_by = self._cooldown_ok(st, now)
        if not ok:
            st.last_pass_blocked_by = blocked_by
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

    # ----------------------------
    # Event consumption
    # ----------------------------
    def consume_passed(self, track_id: int) -> Optional[dict]:
        return self._just_passed.pop(int(track_id), None)

    def pop_any_passed(self) -> Optional[dict]:
        for k in list(self._just_passed.keys()):
            return self._just_passed.pop(k)
        return None
