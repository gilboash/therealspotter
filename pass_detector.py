# pass_detector.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

def _center(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _area(b: Tuple[int,int,int,int]) -> float:
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

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

    passed_time: float = 0.0

class PassDetector:
    """
    Vision-only gate pass detector, based on tracked boxes.
    Works with your Track objects (needs bbox + locked_type + score_ema).

    Call:
      pd.update(tracks, now, frame_w, frame_h)

    Then:
      pd.consume_passed(track_id) to query "just passed" events
      pd.passed_event  (last event dict)  or you can pull per-track state
    """

    def __init__(
        self,
        min_track_score: float = 0.22,
        min_area_ratio: float = 0.030,     # gate is "close enough"
        center_tol: float = 0.18,          # normalized distance to center for "aligned"
        disappear_timeout: float = 0.25,   # seconds after alignment to count disappearance as pass
        flag_cross_min_frac: float = 0.45, # flag center should span this fraction of width to count as wrap
    ):
        self.min_track_score = float(min_track_score)
        self.min_area_ratio = float(min_area_ratio)
        self.center_tol = float(center_tol)
        self.disappear_timeout = float(disappear_timeout)
        self.flag_cross_min_frac = float(flag_cross_min_frac)

        self.states: Dict[int, TrackPassState] = {}
        self._just_passed: Dict[int, dict] = {}

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
            center_dist = (dx*dx + dy*dy) ** 0.5

            area_ratio = a / max(frame_area, 1.0)

            if _is_flag(ttype):
                # "wrap" heuristic: gate center spans left->right over time
                st.min_cx = min(st.min_cx, nx)
                st.max_cx = max(st.max_cx, nx)

                # require being close-ish at some point + a wide horizontal span
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
                # wait for disappearance; handled below by timeout on unseen
                pass

        # handle tracks that disappeared: if aligned recently => PASS
        for tid, st in list(self.states.items()):
            if tid in seen_ids:
                continue

            # disappeared
            if st.stage == "aligned":
                if (now - st.last_seen_time) <= self.disappear_timeout:
                    # disappeared quickly after alignment => likely passed through
                    self._mark_passed(st, now, reason="disappear_after_align")

            # cleanup old states
            if (now - st.last_seen_time) > 2.0:
                self.states.pop(tid, None)

    def _mark_passed(self, st: TrackPassState, now: float, reason: str):
        if st.stage == "passed":
            return
        st.stage = "passed"
        st.passed_time = now
        self._just_passed[st.track_id] = {
            "track_id": st.track_id,
            "type": st.ttype,
            "time": now,
            "reason": reason,
        }

    def consume_passed(self, track_id: int) -> Optional[dict]:
        return self._just_passed.pop(int(track_id), None)

    def pop_any_passed(self) -> Optional[dict]:
        # returns one event if exists
        for k in list(self._just_passed.keys()):
            return self._just_passed.pop(k)
        return None
