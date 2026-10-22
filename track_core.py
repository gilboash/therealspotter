"""
track_core.py — the gate tracker and small box helpers, without heavy imports (no torch / CLIP / Ultralytics).

Shared by lazy_spotter.py (which re-exports these names) and the live spotter, which also runs on the media box.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


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
    def __init__(
        self,
        iou_match_thresh: float = 0.05,
        ttl_seconds: float = 0.03,
        lock_min_score: float = 0.2,
        lock_hysteresis: float = 0.1,
        lock_streak: int = 3,
        ema_alpha: float = 0.4,
        min_det_conf: float = 0.0,
    ):
        self.iou_match_thresh = iou_match_thresh
        self.ttl_seconds = ttl_seconds
        self.lock_min_score = lock_min_score
        self.lock_hysteresis = lock_hysteresis
        self.lock_streak = lock_streak
        self.ema_alpha = ema_alpha
        self.min_det_conf = min_det_conf

        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, dets: List[dict], now: float) -> List[Track]:
        if self.min_det_conf > 0.0:
            dets = [d for d in dets if float(d.get("det_conf", 0.0)) >= self.min_det_conf]

        # Build all valid (score, di, ti) pairs above IoU threshold
        pairs: List[Tuple[float, int, int]] = []
        for di, d in enumerate(dets):
            bb = d["bbox"]
            det_type = d.get("type", "NONE")
            dx1, dy1, dx2, dy2 = bb
            d_area = max(1, (dx2 - dx1) * (dy2 - dy1))
            d_diag = ((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2) ** 0.5 or 1.0
            dcx = (dx1 + dx2) / 2
            dcy = (dy1 + dy2) / 2
            for ti, tr in enumerate(self._tracks):
                if det_type != "NONE" and tr.locked_type != "NONE" and det_type != tr.locked_type:
                    continue
                if iou(bb, tr.bbox) < self.iou_match_thresh:
                    continue
                tx1, ty1, tx2, ty2 = tr.bbox
                t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
                center_dist = ((dcx - (tx1 + tx2) / 2) ** 2 + (dcy - (ty1 + ty2) / 2) ** 2) ** 0.5
                size_ratio = min(d_area, t_area) / max(d_area, t_area)
                pairs.append((center_dist / d_diag + (1.0 - size_ratio), di, ti))

        # Greedy assign best-scoring pairs first
        pairs.sort()
        used_dets: set = set()
        used_tracks: set = set()
        assignments: dict = {}
        for _, di, ti in pairs:
            if di in used_dets or ti in used_tracks:
                continue
            assignments[di] = ti
            used_dets.add(di)
            used_tracks.add(ti)

        # Apply assignments; create new tracks for unmatched detections
        for di, d in enumerate(dets):
            bb = d["bbox"]
            s = float(d.get("type_score", -1.0))
            if di in assignments:
                tr = self._tracks[assignments[di]]
                tr.bbox = bb
                tr.last_seen = now
                tr.score_ema = (1 - self.ema_alpha) * tr.score_ema + self.ema_alpha * max(0.0, s)
                self._update_lock(tr, d.get("type", "NONE"), s)
            else:
                tr = Track(track_id=self._next_id, bbox=bb, last_seen=now)
                self._next_id += 1
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


def _cls_to_name(cls_id: int, names: Dict[int, str]) -> str:
    return names.get(int(cls_id), f"class{int(cls_id)}")


def _crop_padded(frame: np.ndarray, bbox: list, pad_frac: float = 0.5):
    """Crop frame to bbox expanded by pad_frac on each side. Returns (crop, padded_bbox). (As extract_race.py.)"""
    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(W, x2 + px), min(H, y2 + py)
    return frame[ny1:ny2, nx1:nx2], [nx1, ny1, nx2, ny2]
