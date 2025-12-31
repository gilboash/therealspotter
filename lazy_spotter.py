# vocab_spotter.py
import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Your separate pass detector module
from pass_detector import PassDetector

# ============================================================
# DEBUG VISUALIZATION SWITCH
# ============================================================
DEBUG_PASS_VIZ = False   # <-- set False to disable all pass-debug drawing
DEBUG_PASS_PANEL_VIZ = True

PASS_DEBUG_PANEL_WIDTH = 1040            # pixels
PASS_DEBUG_PANEL_BG = (18, 18, 18)      # dark background
PASS_DEBUG_PANEL_TEXT = (235, 235, 235)
PASS_DEBUG_PANEL_DIM = (170, 170, 170)
PASS_DEBUG_PANEL_WARN = (0, 255, 255)
PASS_DEBUG_PANEL_GOOD = (0, 255, 0)

# ============================================================
# Utilities
# ============================================================

def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = areaA + areaB - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


# ============================================================
# Tracking + locking (TIME-BASED TTL)
# ============================================================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float
    locked_type: str = "NONE"
    locked_score: float = -1.0
    type_streak: int = 0
    score_ema: float = 0.0


class TimeTracker:
    """
    IOU tracker with type locking + TIME-BASED expiration.
    - A track expires if (now - last_seen) > ttl_seconds
    """

    def __init__(
        self,
        iou_match_thresh: float = 0.3,
        ttl_seconds: float = 0.6,
        lock_min_score: float = 0.20,
        lock_hysteresis: float = 0.10,
        lock_streak: int = 3,
        ema_alpha: float = 0.4,
    ):
        self.iou_match_thresh = iou_match_thresh
        self.ttl_seconds = ttl_seconds
        self.lock_min_score = lock_min_score
        self.lock_hysteresis = lock_hysteresis
        self.lock_streak = lock_streak
        self.ema_alpha = ema_alpha

        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, dets: List[dict], now: float) -> List[Track]:
        used = set()

        for d in dets:
            bb = d["bbox"]
            best_iou = 0.0
            best_ti = None
            for ti, tr in enumerate(self._tracks):
                if ti in used:
                    continue
                s = iou(bb, tr.bbox)
                if s > best_iou:
                    best_iou = s
                    best_ti = ti

            if best_ti is not None and best_iou >= self.iou_match_thresh:
                tr = self._tracks[best_ti]
                used.add(best_ti)

                tr.bbox = bb
                tr.last_seen = now

                s = float(d.get("type_score", -1.0))
                tr.score_ema = (1 - self.ema_alpha) * tr.score_ema + self.ema_alpha * max(0.0, s)
                self._update_lock(tr, d.get("type", "NONE"), s)
            else:
                tr = Track(track_id=self._next_id, bbox=bb, last_seen=now)
                self._next_id += 1
                s = float(d.get("type_score", -1.0))
                tr.score_ema = max(0.0, s)
                self._update_lock(tr, d.get("type", "NONE"), s, is_new=True)
                self._tracks.append(tr)

        self._tracks = [t for t in self._tracks if (now - t.last_seen) <= self.ttl_seconds]
        return list(self._tracks)

    def _update_lock(self, tr: Track, new_type: str, new_score: float, is_new: bool = False):
        if new_type is None:
            new_type = "NONE"
        new_score = float(new_score)

        if is_new and new_score >= self.lock_min_score and new_type != "NONE":
            tr.locked_type = new_type
            tr.locked_score = new_score
            tr.type_streak = 1
            return

        if tr.locked_type == "NONE":
            if new_type != "NONE" and new_score >= self.lock_min_score:
                tr.locked_type = new_type
                tr.locked_score = new_score
                tr.type_streak = 1
            return

        if new_type == tr.locked_type:
            tr.locked_score = max(tr.locked_score, new_score)
            tr.type_streak = min(tr.type_streak + 1, 999)
            return

        if new_type != "NONE" and (new_score >= tr.locked_score + self.lock_hysteresis):
            tr.type_streak += 1
            if tr.type_streak >= self.lock_streak:
                tr.locked_type = new_type
                tr.locked_score = new_score
                tr.type_streak = 1
        else:
            tr.type_streak = 0


# ============================================================
# Visualization
# ============================================================

DEFAULT_COLORS = {
    "square": (255, 0, 0),     # blue
    "circle": (0, 255, 0),     # green
    "arch": (0, 255, 255),     # yellow
    "flagpole": (255, 0, 255), # magenta
    "gate": (0, 165, 255),     # orange (generic)
    "NONE": (128, 128, 128),   # gray
}

def type_color(type_name: str, palette: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    return palette.get(type_name, (0, 165, 255))  # orange default

def draw_legend(frame: np.ndarray, types: List[str], palette: Dict[str, Tuple[int, int, int]], show_none: bool):
    x, y = 10, 20
    cv2.putText(frame, "Legend:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    y += 20
    for t in types:
        if (not show_none) and t == "NONE":
            continue
        c = type_color(t, palette)
        cv2.rectangle(frame, (x, y - 12), (x + 14, y + 2), c, -1)
        cv2.putText(frame, t, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
        y += 20


# ----------------------------
# NEW: pass-detector debug viz
# ----------------------------

def _norm_type(s: str) -> str:
    return (s or "").strip().lower()

def _is_flag(t: str) -> bool:
    t = _norm_type(t)
    return ("flag" in t) or ("pole" in t)

def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _put_text_lines(img, x, y, lines, font_scale=0.45, thickness=1, color=(240, 240, 240), line_h=16):
    yy = y
    for ln in lines:
        cv2.putText(img, ln, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        yy += line_h

def _alpha_rect(img, x1, y1, x2, y2, alpha=0.55, color=(0, 0, 0)):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

def _cooldown_remaining(passdet: PassDetector, track_id: int, ttype: str, now: float) -> Tuple[float, float, float]:
    """
    Visualization only: compute remaining cooldowns (global/type/track) based on passdet internals.
    """
    # These exist in your pass_detector.py; if not, just return zeros
    g_last = getattr(passdet, "_last_pass_time_global", -1e9)
    by_type = getattr(passdet, "_last_pass_time_by_type", {})
    by_track = getattr(passdet, "_last_pass_time_by_track", {})

    pass_cd = float(getattr(passdet, "pass_cooldown_sec", 0.0))
    type_cd = float(getattr(passdet, "type_cooldown_sec", 0.0))
    track_cd = float(getattr(passdet, "track_cooldown_sec", 0.0))

    rem_g = max(0.0, pass_cd - (now - float(g_last)))
    tkey = _norm_type(ttype)
    rem_t = max(0.0, type_cd - (now - float(by_type.get(tkey, -1e9))))
    rem_tr = max(0.0, track_cd - (now - float(by_track.get(int(track_id), -1e9))))
    return rem_g, rem_t, rem_tr

def build_pass_debug_panel(
    frame_h: int,
    frame_w: int,
    now: float,
    tracks: List[Track],
    passdet: PassDetector,
    title: str = "PassDetector Debug",
) -> np.ndarray:
    """
    Renders a table-like panel image showing PassDetector internal state per track_id.
    Uses passdet.states + current Track list (for score_ema).
    """
    panel_w = PASS_DEBUG_PANEL_WIDTH
    panel = np.zeros((frame_h, panel_w, 3), dtype=np.uint8)
    panel[:, :] = PASS_DEBUG_PANEL_BG

    # maps for easy join
    tr_by_id: Dict[int, Track] = {int(t.track_id): t for t in tracks}
    st_by_id = getattr(passdet, "states", {}) or {}

    # header
    y = 26
    cv2.putText(panel, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, PASS_DEBUG_PANEL_TEXT, 2, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, f"t={now:.3f}s  states={len(st_by_id)}  tracks={len(tracks)}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PASS_DEBUG_PANEL_DIM, 2, cv2.LINE_AA)
    y += 20

    # column header
    col = "tid  type       stg     score   area%   cdist   seenAgo  misc"
    cv2.putText(panel, col, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, PASS_DEBUG_PANEL_TEXT, 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (10, y), (panel_w - 10, y), (70, 70, 70), 1)
    y += 18

    frame_area = float(max(1, frame_w * frame_h))

    # sort by track_id for stable reading
    items = sorted(st_by_id.items(), key=lambda kv: int(kv[0]))

    line_h = 18
    max_lines = (frame_h - y - 10) // line_h

    # show only first N lines if too many
    if len(items) > max_lines:
        items = items[:max_lines]
        cv2.putText(panel, f"(showing first {max_lines} of {len(st_by_id)})",
                    (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PASS_DEBUG_PANEL_WARN, 2, cv2.LINE_AA)

    for tid, st in items:
        tid = int(tid)
        tr = tr_by_id.get(tid)

        # st fields exist in your pass_detector TrackPassState
        ttype = str(getattr(st, "ttype", ""))
        stage = str(getattr(st, "stage", ""))
        last_area = float(getattr(st, "last_area", 0.0))
        last_cx = float(getattr(st, "last_cx", 0.0))
        last_cy = float(getattr(st, "last_cy", 0.0))
        last_seen_time = float(getattr(st, "last_seen_time", 0.0))

        area_ratio = last_area / frame_area
        nx = last_cx / max(frame_w, 1)
        ny = last_cy / max(frame_h, 1)
        cdist = ((abs(nx - 0.5) ** 2 + abs(ny - 0.5) ** 2) ** 0.5)
        seen_ago = max(0.0, now - last_seen_time)

        score = float(tr.score_ema) if tr is not None else -1.0

        # misc for flag logic
        min_cx = float(getattr(st, "min_cx", 0.0))
        max_cx = float(getattr(st, "max_cx", 0.0))
        span = max_cx - min_cx

        # color cues
        color = PASS_DEBUG_PANEL_TEXT
        if stage == "aligned":
            color = PASS_DEBUG_PANEL_WARN
        if stage == "passed":
            color = PASS_DEBUG_PANEL_GOOD

        short_type = (ttype[:9] + "…") if len(ttype) > 10 else ttype.ljust(10)
        short_stage = stage.ljust(7)[:7]

        misc = f"span={span:.2f}"
        line = f"{tid:>3d}  {short_type} {short_stage} {score:>5.2f}  {area_ratio*100:>5.2f}  {cdist:>6.3f}  {seen_ago:>6.2f}  {misc}"

        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        y += line_h

    return panel

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    palette: Dict[str, Tuple[int, int, int]],
    show_none: bool,
    hide_after: float,
    now: float,
    passhud: Optional["PassHUD"] = None,
    passdet: Optional[PassDetector] = None,   # <-- NEW (viz only)
):
    H, W = frame.shape[:2]
    frame_area = float(W * H)

    # Map state for quick lookup (viz only)
    st_map = getattr(passdet, "states", {}) if passdet is not None else {}

    for tr in tracks:
        t = tr.locked_type if tr.locked_type else "NONE"
        if (not show_none) and t == "NONE":
            continue
        if hide_after > 0 and (now - tr.last_seen) > hide_after:
            continue

        x1, y1, x2, y2 = tr.bbox
        c = type_color(t, palette)

        highlighted = (passhud is not None) and passhud.is_highlighted(tr.track_id, now)

        thick = 6 if highlighted else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick)

        label = f"#{tr.track_id} {t} {tr.score_ema:.2f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)

        # ---- NEW: per-box pass debug (stage/area/center/cooldowns) ----
        if DEBUG_PASS_VIZ and passdet is not None and int(tr.track_id) in st_map:
            st = st_map[int(tr.track_id)]
            stage = str(getattr(st, "stage", ""))
            la = float(getattr(st, "last_area", 0.0))
            ar = la / max(frame_area, 1.0)

            cx, cy = _center(tr.bbox)
            nx = cx / max(W, 1)
            ny = cy / max(H, 1)
            dx = abs(nx - 0.5)
            dy = abs(ny - 0.5)
            center_dist = (dx * dx + dy * dy) ** 0.5

            rem_g, rem_t, rem_tr = _cooldown_remaining(passdet, int(tr.track_id), str(getattr(st, "ttype", "")), now)
            dbg1 = f"stage={stage}  area={ar:.3f}  cDist={center_dist:.3f}"
            dbg2 = f"cd(g/t/tr)={rem_g:.2f}/{rem_t:.2f}/{rem_tr:.2f}"

            # small black background just above bbox (or inside top)
            box_w = max(120, min(320, x2 - x1))
            bx1 = x1
            by2 = max(0, y1 - 2)
            by1 = max(0, by2 - 34)
            _alpha_rect(frame, bx1, by1, bx1 + box_w, by2, alpha=0.55, color=(0, 0, 0))
            cv2.putText(frame, dbg1, (bx1 + 3, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, dbg2, (bx1 + 3, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        if highlighted:
            cv2.putText(
                frame,
                "PASSED",
                (x1, min(frame.shape[0] - 10, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


# ============================================================
# PASSED HUD (persistent list)
# ============================================================

@dataclass
class PassEvent:
    idx: int
    track_id: int
    type: str
    reason: str
    t: float

# ============================================================
# PASSED HUD (persistent list) + highlight support
# ============================================================

@dataclass
class PassEvent:
    idx: int
    track_id: int
    type: str
    reason: str
    t: float

class PassHUD:
    def __init__(self, keep_seconds: float = 3.0, max_events: int = 8, highlight_seconds: float = 1.2):
        self.keep_seconds = float(keep_seconds)
        self.max_events = int(max_events)
        self.highlight_seconds = float(highlight_seconds)

        self._events: List[PassEvent] = []
        self._next_idx = 1

        # track_id -> highlight-until timestamp
        self._highlight_until: Dict[int, float] = {}

    def add(self, now: float, track_id: int, gate_type: str, reason: str):
        self._events.append(
            PassEvent(
                idx=self._next_idx,
                track_id=int(track_id),
                type=str(gate_type),
                reason=str(reason),
                t=float(now),
            )
        )
        self._next_idx += 1

        # highlight this specific track for a short time
        self._highlight_until[int(track_id)] = now + self.highlight_seconds

        # cap
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def _prune(self, now: float):
        if self.keep_seconds > 0:
            cutoff = now - self.keep_seconds
            self._events = [e for e in self._events if e.t >= cutoff]

        # prune highlight map too
        for tid in list(self._highlight_until.keys()):
            if now >= self._highlight_until[tid]:
                self._highlight_until.pop(tid, None)

    def is_highlighted(self, track_id: int, now: float) -> bool:
        t = self._highlight_until.get(int(track_id), 0.0)
        return now < t

    def draw(self, frame: np.ndarray, now: float, x: int = 10, y: int = 60, align_right: bool = False, margin: int = 10):
        self._prune(now)
        if not self._events:
            return

        # Right-align option
        if align_right:
            h, w = frame.shape[:2]
            lines = [f"{e.idx}) #{e.track_id} {e.type}" for e in self._events]
            longest = max(lines, key=len)
            est_px = int(len(longest) * 13)
            x = max(margin, w - est_px - margin)

        cv2.putText(frame, "PASSED:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26

        for e in reversed(self._events):  # newest first
            line = f"{e.idx}) #{e.track_id} {e.type}  ({e.reason})"
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
            y += 22


# ============================================================
# YOLO multiclass helpers (NEW detector path)
# ============================================================

def _get_yolo_names(det: YOLO) -> Dict[int, str]:
    names = getattr(det.model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}

def _cls_to_name(cls_id: int, names: Dict[int, str]) -> str:
    return names.get(int(cls_id), f"class{int(cls_id)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FPV gate spotter: multiclass YOLO detector + type-lock tracking + optional PASS HUD"
    )

    parser.add_argument("--mode", choices=["calib", "learn", "race"], default="learn")
    parser.add_argument("--video", type=str, default=None, help="Path to MP4 file (optional). If omitted, uses camera 0.")

    # kept for backwards compatibility (unused now)
    parser.add_argument("--vocab", type=str, default=None, help="(unused) old vocab folder arg; safe to omit now")
    parser.add_argument("--device", type=str, default="cpu", help="Embedding device (kept for compatibility)")

    parser.add_argument("--show-none", action="store_true", help="Visualize NONE-class tracks (default: hidden)")
    parser.add_argument("--max-candidates", type=int, default=10, help="Max detections per frame after filtering/sort")
    parser.add_argument("--min-type-score", type=float, default=0.20, help="Min detector confidence to accept class; else NONE")

    # Stage 1 detector config (NOW: your trained multiclass YOLO model)
    parser.add_argument("--det-model", type=str, required=True, help="Path to your trained YOLO .pt model")
    parser.add_argument("--det-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--det-maxdet", type=int, default=50, help="YOLO max detections per frame")

    # Tracking + locking knobs
    parser.add_argument("--iou-match", type=float, default=0.3, help="IOU association threshold")
    parser.add_argument("--ttl-seconds", type=float, default=0.3, help="Track TTL in seconds")
    parser.add_argument("--hide-after", type=float, default=0.1, help="Hide tracks if not seen for this many seconds (0 disables)")
    parser.add_argument("--lock-min-score", type=float, default=0.20)
    parser.add_argument("--lock-hysteresis", type=float, default=0.10)
    parser.add_argument("--lock-streak", type=int, default=3)

    # Pass detector
    parser.add_argument("--pass-enable", action="store_true", help="Enable gate pass detection")
    parser.add_argument("--pass-min-score", type=float, default=0.22, help="Min track score_ema for pass logic")
    parser.add_argument("--pass-min-area", type=float, default=0.030, help="Min bbox area ratio (gate close enough)")
    parser.add_argument("--pass-center-tol", type=float, default=0.18, help="Center tolerance for alignment (0..1)")
    parser.add_argument("--pass-disappear", type=float, default=0.25, help="Seconds after alignment where disappearance counts as pass")
    parser.add_argument("--pass-hud-seconds", type=float, default=3.0, help="How long to keep PASSED lines on screen")
    parser.add_argument("--pass-hud-max", type=int, default=8, help="Max PASSED lines shown")

    args = parser.parse_args()

    det = YOLO(args.det_model)
    names = _get_yolo_names(det)

    tracker = TimeTracker(
        iou_match_thresh=args.iou_match,
        ttl_seconds=args.ttl_seconds,
        lock_min_score=args.lock_min_score,
        lock_hysteresis=args.lock_hysteresis,
        lock_streak=args.lock_streak,
    )

    passdet: Optional[PassDetector] = None
    passhud: Optional[PassHUD] = None
    if args.pass_enable:
        passdet = PassDetector(
            min_track_score=args.pass_min_score,
            min_area_ratio=args.pass_min_area,
            center_tol=args.pass_center_tol,
            disappear_timeout=args.pass_disappear,
            ignore_flagpoles=False,  # <--- your current setting
        )
        passhud = PassHUD(keep_seconds=args.pass_hud_seconds, max_events=args.pass_hud_max)

    palette = dict(DEFAULT_COLORS)
    for _, n in names.items():
        palette.setdefault(str(n), (0, 165, 255))  # orange default

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")

    print(f"Detector model: {args.det_model}")
    print(f"Classes: {list(names.values()) if names else '(unknown names)'}")
    if args.pass_enable:
        print("Pass detector: ENABLED")
    print("Press 'q' to quit.")
    print("Controls: SPACE pause/resume | n step (when paused)")

    last_t = 0.0

    paused = False
    step_once = False
    frame = None

    while True:
        if not paused or step_once:
            ok, frame = cap.read()
            step_once = False
            if not ok:
                break

        # Use video timestamps as "now" for pass logic (good for frame-by-frame debugging)
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        now = pos_msec / 1000.0

        dt = now - last_t
        last_t = now

        H, W = frame.shape[:2]

        # multiclass YOLO inference
        res = det(frame, conf=args.det_conf, verbose=False, max_det=args.det_maxdet)[0]

        typed: List[dict] = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1, y1, x2, y2), W, H)

            conf = float(b.conf[0]) if b.conf is not None else 0.0
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            cls_name = _cls_to_name(cls_id, names)

            gate_type = cls_name if conf >= args.min_type_score else "NONE"

            typed.append({
                "bbox": bb,
                "det_conf": conf,
                "type": gate_type,
                "type_score": conf,
            })

        typed_sorted = sorted(typed, key=lambda d: d["det_conf"], reverse=True)
        typed_topk = typed_sorted[:max(1, args.max_candidates)]

        tracks = tracker.update(typed_topk, now)

        if passdet is not None:
            passdet.update(tracks, now, frame_w=W, frame_h=H)
            while True:
                evt = passdet.pop_any_passed()
                if evt is None:
                    break
                if passhud is not None:
                    passhud.add(
                        now,
                        track_id=evt.get("track_id", -1),
                        gate_type=evt.get("type", "UNKNOWN"),
                        reason=evt.get("reason", ""),
                    )

        if args.mode in ("calib", "learn", "race"):
            draw_tracks(
                frame,
                tracks,
                palette=palette,
                show_none=args.show_none,
                hide_after=args.hide_after,
                now=now,
                passhud=passhud,
                passdet=passdet,   # <-- NEW (viz only)
            )

            legend_types = list(dict.fromkeys([str(n) for n in names.values()])) if names else ["gate"]
            if args.show_none:
                legend_types.append("NONE")
            draw_legend(frame, legend_types, palette, show_none=args.show_none)

            cv2.putText(frame, f"MODE: {args.mode.upper()}  dt={dt*1000:.1f}ms",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Detections={len(typed)} TopK={len(typed_topk)} Tracks={len(tracks)}",
                        (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

            if passhud is not None:
                passhud.draw(frame, now, x=300, y=60)

            # ---- NEW: right-side pass debug panel ----
            if DEBUG_PASS_PANEL_VIZ and passdet is not None:
               
                panel = build_pass_debug_panel(
                        frame_h=H,
                        frame_w=W,
                        now=now,
                        tracks=tracks,
                        passdet=passdet,
                        title="PassDetector Debug",
                    )
                cv2.imshow("PassDetector Debug", panel)
            cv2.imshow("Gate Spotter (YOLO multiclass)", frame)

        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):  # SPACE = toggle pause
            paused = not paused
            print(f"[DEBUG] paused = {paused}")
        elif key == ord("n") and paused:  # next frame
            step_once = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
