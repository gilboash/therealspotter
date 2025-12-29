# vocab_spotter.py
import os
import glob
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Your separate pass detector module
from pass_detector import PassDetector

# SSL cert helper for open_clip downloads (if used)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

_CLIP_AVAILABLE = False
try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except Exception:
    _CLIP_AVAILABLE = False


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


def crop_with_padding(frame: np.ndarray, bbox: Tuple[int, int, int, int], pad_frac: float = 0.10) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_frac)
    py = int(bh * pad_frac)
    bb = clamp_bbox((x1 - px, y1 - py, x2 + px, y2 + py), w, h)
    x1, y1, x2, y2 = bb
    return frame[y1:y2, x1:x2].copy()


# ============================================================
# Stage 2: Vocabulary embedder (CLIP or histogram fallback)
# ============================================================

class VocabEmbedder:
    def __init__(self, vocab_dir: str, device: str = "cpu", clip_model: str = "ViT-B-32", clip_pretrained: str = "openai"):
        self.vocab_dir = vocab_dir
        self.device = device
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained

        self.available_types: List[str] = []
        self.type_to_proto: Dict[str, np.ndarray] = {}

        self.backend = "clip" if _CLIP_AVAILABLE else "hist"
        self._clip_model = None
        self._clip_preprocess = None
        if self.backend == "clip":
            self._init_clip()

        self._build_prototypes()

    def _init_clip(self):
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_model, pretrained=self.clip_pretrained
        )
        self._clip_model = self._clip_model.to(self.device)
        self._clip_model.eval()

    def _iter_images(self, type_name: str) -> List[str]:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.vocab_dir, type_name, e)))
        return sorted(files)

    def _embed_clip(self, bgr: np.ndarray) -> np.ndarray:
        import PIL.Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)
        tensor = self._clip_preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _embed_hist(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None).flatten().astype(np.float32)
        hist /= (np.linalg.norm(hist) + 1e-9)
        return hist.astype(np.float32)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        return self._embed_clip(bgr) if self.backend == "clip" else self._embed_hist(bgr)

    def _build_prototypes(self):
        types = [d for d in os.listdir(self.vocab_dir) if os.path.isdir(os.path.join(self.vocab_dir, d))]
        types = sorted(types)

        for t in types:
            imgs = self._iter_images(t)
            if not imgs:
                continue
            feats = []
            for p in imgs:
                im = cv2.imread(p)
                if im is None:
                    continue
                feats.append(self.embed(im))
            if not feats:
                continue
            proto = np.mean(np.stack(feats, axis=0), axis=0)
            proto /= (np.linalg.norm(proto) + 1e-9)
            self.type_to_proto[t] = proto.astype(np.float32)
            self.available_types.append(t)

        if not self.available_types:
            raise RuntimeError(
                f"No vocabulary types found in {self.vocab_dir}. "
                f"Expected subfolders like square/, circle/, arch/, flagpole/"
            )

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        feat = self.embed(crop_bgr)
        best_t = "NONE"
        best_s = -1.0
        scores: Dict[str, float] = {}

        for t in self.available_types:
            s = float(np.dot(feat, self.type_to_proto[t]))
            scores[t] = s
            if s > best_s:
                best_s = s
                best_t = t

        return best_t, best_s, scores


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
    "NONE": (128, 128, 128),   # gray
}

def type_color(type_name: str, palette: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    return palette.get(type_name, (0, 165, 255))  # orange for unknown types

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

def draw_tracks(frame: np.ndarray, tracks: List[Track], palette: Dict[str, Tuple[int, int, int]], show_none: bool, hide_after: float, now: float):
    for tr in tracks:
        t = tr.locked_type if tr.locked_type else "NONE"
        if (not show_none) and t == "NONE":
            continue
        if hide_after > 0 and (now - tr.last_seen) > hide_after:
            continue

        x1, y1, x2, y2 = tr.bbox
        c = type_color(t, palette)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
        label = f"#{tr.track_id} {t} {tr.score_ema:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


# ============================================================
# PASSED HUD (persistent list)
# ============================================================

@dataclass
class PassEvent:
    idx: int
    type: str
    reason: str
    t: float

class PassHUD:
    def __init__(self, keep_seconds: float = 3.0, max_events: int = 8):
        self.keep_seconds = float(keep_seconds)
        self.max_events = int(max_events)
        self._events: List[PassEvent] = []
        self._next_idx = 1

    def add(self, now: float, gate_type: str, reason: str):
        self._events.append(PassEvent(idx=self._next_idx, type=str(gate_type), reason=str(reason), t=float(now)))
        self._next_idx += 1
        # cap
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def _prune(self, now: float):
        if self.keep_seconds <= 0:
            return
        cutoff = now - self.keep_seconds
        self._events = [e for e in self._events if e.t >= cutoff]

    def draw(self, frame: np.ndarray, now: float, x: int = 10, y: int = 60):
        self._prune(now)

        if not self._events:
            return

        cv2.putText(frame, "PASSED:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26

        for e in reversed(self._events):  # newest first
            line = f"{e.idx}) {e.type}  ({e.reason})"
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            y += 22


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage FPV gate spotter: permissive proposals + vocab matching + type-lock tracking + PASS HUD"
    )

    parser.add_argument("--mode", choices=["calib", "learn", "race"], default="learn")
    parser.add_argument("--video", type=str, default=None, help="Path to MP4 file (optional). If omitted, uses camera 0.")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary folder with subfolders per type")
    parser.add_argument("--device", type=str, default="cpu", help="Embedding device: cpu or mps (Mac) or cuda")

    parser.add_argument("--show-none", action="store_true", help="Visualize NONE-class tracks (default: hidden)")
    parser.add_argument("--max-candidates", type=int, default=4, help="Max candidates per frame (top by type score)")
    parser.add_argument("--min-type-score", type=float, default=0.20, help="Min similarity score to accept type; else NONE")

    # Stage 1 detector config (general permissive detector)
    parser.add_argument("--det-model", type=str, default="yolov8n.pt", help="YOLO model for permissive proposals (general)")
    parser.add_argument("--det-conf", type=float, default=0.08, help="YOLO confidence threshold (permissive)")
    parser.add_argument("--det-maxdet", type=int, default=30, help="YOLO max detections per frame before our top-K filtering")

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

    embedder = VocabEmbedder(vocab_dir=args.vocab, device=args.device)
    det = YOLO(args.det_model)

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
        )
        passhud = PassHUD(keep_seconds=args.pass_hud_seconds, max_events=args.pass_hud_max)

    palette = dict(DEFAULT_COLORS)
    for t in embedder.available_types:
        palette.setdefault(t, (0, 165, 255))  # orange for unknown

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")

    print(f"Backend: {embedder.backend} | Types: {embedder.available_types}")
    if args.pass_enable:
        print("Pass detector: ENABLED")
    print("Press 'q' to quit.")

    last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_t
        last_t = now

        H, W = frame.shape[:2]

        # Stage 1 proposals
        res = det(frame, conf=args.det_conf, verbose=False, max_det=args.det_maxdet)[0]
        proposals = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1, y1, x2, y2), W, H)
            proposals.append({"bbox": bb, "det_conf": float(b.conf[0])})

        # Stage 2 vocab matching
        typed = []
        for p in proposals:
            crop = crop_with_padding(frame, p["bbox"], pad_frac=0.10)
            best_t, best_s, _ = embedder.classify(crop)
            if best_s < args.min_type_score:
                best_t = "NONE"
            typed.append({
                "bbox": p["bbox"],
                "det_conf": p["det_conf"],
                "type": best_t,
                "type_score": float(best_s),
            })

        # Top-K candidates by type_score then det_conf
        typed_sorted = sorted(typed, key=lambda d: (d["type_score"], d["det_conf"]), reverse=True)
        typed_topk = typed_sorted[:max(1, args.max_candidates)]

        # Update tracker
        tracks = tracker.update(typed_topk, now)

        # Pass detector (agnostic, no expected-order coupling)
        if passdet is not None:
            passdet.update(tracks, now, frame_w=W, frame_h=H)
            # Drain all pass events this frame (in case it emits multiple)
            while True:
                evt = passdet.pop_any_passed()
                if evt is None:
                    break
                # expected keys: evt["type"], evt["reason"]
                if passhud is not None:
                    passhud.add(now, gate_type=evt.get("type", "UNKNOWN"), reason=evt.get("reason", ""))

        # Visualization
        if args.mode in ("calib", "learn", "race"):
            draw_tracks(frame, tracks, palette=palette, show_none=args.show_none, hide_after=args.hide_after, now=now)

            # in calib/learn, show faint proposals
            if args.mode in ("calib", "learn"):
                for p in proposals:
                    x1, y1, x2, y2 = p["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

            # legend
            legend_types = list(embedder.available_types)
            if args.show_none:
                legend_types.append("NONE")
            draw_legend(frame, legend_types, palette, show_none=args.show_none)

            # HUD
            cv2.putText(frame, f"MODE: {args.mode.upper()}  dt={dt*1000:.1f}ms",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Proposals={len(proposals)} TopK={len(typed_topk)} Tracks={len(tracks)}",
                        (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

            # PASSED list (persistent)
            if passhud is not None:
                passhud.draw(frame, now, x=10, y=60)

            cv2.imshow("Vocab Gate Spotter", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
