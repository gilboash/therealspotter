import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

# =========================================================
# Config
# =========================================================

TYPE_COLORS = {
    "square":   (255, 180, 0),   # light blue-ish (BGR)
    "circle":   (0, 255, 255),   # yellow
    "arch":     (255, 0, 255),   # magenta
    "flagpole": (0, 165, 255),   # orange
    "none":     (160, 160, 160), # gray
}

ALLOWED_TYPES = ["square", "circle", "arch", "flagpole"]

# Stage-1 permissive detector defaults
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.30          # permissive
YOLO_IOU_NMS = 0.50       # NMS IoU

# Stage-2 vocab matching
ORB_NFEATURES = 1000
GOOD_MATCH_DIST = 50      # lower => stricter matches
MIN_GOOD_MATCHES = 12     # below => "none"
TYPE_SWITCH_MARGIN = 0.10 # how much better a new type must be to override current
TYPE_SWITCH_FRAMES = 3    # require consecutive frames before switching type

# Tracking
TRACK_IOU_THRESH = 0.30
TRACK_MAX_MISSES = 12

# =========================================================
# Utilities
# =========================================================

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, ax2-ax1) * max(0, ay2-ay1)
    areaB = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / float(areaA + areaB - inter + 1e-9)

def clamp_bbox(b, w, h):
    x1,y1,x2,y2 = b
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return (x1,y1,x2,y2)

def draw_legend(frame):
    y = 20
    cv2.putText(frame, "Legend:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)
    y += 22
    for t in ALLOWED_TYPES + ["none"]:
        color = TYPE_COLORS.get(t, (200,200,200))
        cv2.rectangle(frame, (10, y-12), (30, y+4), color, -1)
        cv2.putText(frame, t, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
        y += 22

# =========================================================
# Vocabulary embedder (ORB-based; no downloads needed)
# =========================================================

class VocabMatcher:
    def __init__(self, vocab_dir: str):
        self.vocab_dir = vocab_dir
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db: Dict[str, List[np.ndarray]] = {t: [] for t in ALLOWED_TYPES}
        self._load_vocab()

    def _load_vocab(self):
        if not os.path.isdir(self.vocab_dir):
            raise ValueError(f"vocab_dir not found: {self.vocab_dir}")

        for t in ALLOWED_TYPES:
            tdir = os.path.join(self.vocab_dir, t)
            if not os.path.isdir(tdir):
                continue

            for fn in os.listdir(tdir):
                if not fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    continue
                path = os.path.join(tdir, fn)
                img = cv2.imread(path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = self.orb.detectAndCompute(gray, None)
                if des is not None and len(des) > 0:
                    self.db[t].append(des)

        # Basic sanity
        loaded = {t: len(self.db[t]) for t in ALLOWED_TYPES}
        if sum(loaded.values()) == 0:
            raise ValueError(f"No vocab images loaded from {self.vocab_dir}. Expected subfolders: {ALLOWED_TYPES}")
        print(f"[Vocab] Loaded descriptors: {loaded}")

    def classify_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns: (best_type or 'none', confidence 0..1)
        Confidence is based on good-match ratio vs best type.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return ("none", 0.0)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(gray, None)
        if des1 is None or len(des1) < 10:
            return ("none", 0.0)

        best_type = "none"
        best_score = 0.0

        for t, desc_list in self.db.items():
            # compare against all exemplars; take best
            t_best = 0.0
            for des2 in desc_list:
                matches = self.bf.match(des1, des2)
                if not matches:
                    continue
                good = [m for m in matches if m.distance <= GOOD_MATCH_DIST]
                # score normalized by descriptor count (rough)
                denom = max(len(des1), len(des2), 1)
                score = len(good) / denom
                if score > t_best:
                    t_best = score

            if t_best > best_score:
                best_score = t_best
                best_type = t

        # Convert score into a more interpretable confidence
        # Also enforce minimum evidence (good matches)
        # We'll estimate good matches from score * max_desc (approx).
        approx_good = int(best_score * max(len(des1), 1))

        if approx_good < MIN_GOOD_MATCHES:
            return ("none", float(min(best_score * 5.0, 1.0)))  # small signal but reject type

        # map score (usually small ~0.02-0.2) to 0..1
        conf = float(min(best_score * 8.0, 1.0))
        return (best_type, conf)

# =========================================================
# Stage-1 permissive detector (general)
# =========================================================

class PermissiveDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH, conf=YOLO_CONF, iou_nms=YOLO_IOU_NMS):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou_nms = iou_nms

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        # We intentionally ignore classes; treat everything as a proposal.
        res = self.model(frame_bgr, conf=self.conf, iou=self.iou_nms, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            out.append({
                "bbox": (x1,y1,x2,y2),
                "det_conf": float(b.conf[0]),
                "cls": int(b.cls[0])
            })
        return out

# =========================================================
# Tracking + Type locking
# =========================================================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int,int,int,int]
    misses: int = 0
    age: int = 0

    # locked type state
    gate_type: str = "none"
    type_conf: float = 0.0
    pending_type: Optional[str] = None
    pending_hits: int = 0

class IoUTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict]) -> List[Track]:
        # match detections to existing tracks by IoU (greedy)
        unmatched_dets = set(range(len(detections)))
        matched = set()

        # compute IoU matrix
        iou_mat = []
        for t in self.tracks:
            row = []
            for d in detections:
                row.append(iou(t.bbox, d["bbox"]))
            iou_mat.append(row)

        # greedy match: pick highest IoU pairs
        while True:
            best = 0.0
            best_t = -1
            best_d = -1
            for ti, row in enumerate(iou_mat):
                if ti in matched:
                    continue
                for di, v in enumerate(row):
                    if di not in unmatched_dets:
                        continue
                    if v > best:
                        best = v
                        best_t = ti
                        best_d = di
            if best_t == -1 or best < TRACK_IOU_THRESH:
                break

            # assign
            tr = self.tracks[best_t]
            tr.bbox = detections[best_d]["bbox"]
            tr.misses = 0
            tr.age += 1
            matched.add(best_t)
            unmatched_dets.remove(best_d)

        # increment misses on unmatched tracks
        for tr in self.tracks:
            if tr.track_id not in [self.tracks[i].track_id for i in matched]:
                tr.misses += 1
                tr.age += 1

        # drop stale
        self.tracks = [t for t in self.tracks if t.misses <= TRACK_MAX_MISSES]

        # create new tracks for unmatched detections
        for di in list(unmatched_dets):
            tr = Track(track_id=self.next_id, bbox=detections[di]["bbox"])
            self.next_id += 1
            self.tracks.append(tr)

        return self.tracks

def lock_type(track: Track, new_type: str, new_conf: float):
    """
    Stabilize type assignment:
    - if track has no type => accept quickly
    - if new type differs => require it to be consistently better for TYPE_SWITCH_FRAMES
    """
    if new_type == "none" or new_conf <= 0:
        # don't switch to none; keep existing lock unless nothing set
        if track.gate_type == "none":
            track.type_conf = max(track.type_conf * 0.9, new_conf)
        return

    if track.gate_type == "none":
        # first confident assignment
        track.gate_type = new_type
        track.type_conf = new_conf
        track.pending_type = None
        track.pending_hits = 0
        return

    if new_type == track.gate_type:
        # reinforce
        track.type_conf = 0.7 * track.type_conf + 0.3 * new_conf
        track.pending_type = None
        track.pending_hits = 0
        return

    # different type: only switch if clearly better consistently
    if new_conf >= track.type_conf + TYPE_SWITCH_MARGIN:
        if track.pending_type == new_type:
            track.pending_hits += 1
        else:
            track.pending_type = new_type
            track.pending_hits = 1

        if track.pending_hits >= TYPE_SWITCH_FRAMES:
            track.gate_type = new_type
            track.type_conf = new_conf
            track.pending_type = None
            track.pending_hits = 0
    else:
        # decay pending
        if track.pending_type is not None:
            track.pending_hits = max(0, track.pending_hits - 1)
            if track.pending_hits == 0:
                track.pending_type = None

# =========================================================
# Main runner
# =========================================================

def run(vocab_dir: str, video: Optional[str] = None):
    cap = cv2.VideoCapture(0 if video is None else video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")

    detector = PermissiveDetector()
    matcher = VocabMatcher(vocab_dir)
    tracker = IoUTracker()

    print("Controls: q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # Stage 1: permissive proposals
        dets = detector.detect(frame)

        # Update tracker first (so IDs persist even if type is none)
        tracks = tracker.update(dets)

        # Stage 2: classify each tracked crop via vocabulary
        for tr in tracks:
            x1,y1,x2,y2 = clamp_bbox(tr.bbox, W, H)
            crop = frame[y1:y2, x1:x2].copy()

            t, conf = matcher.classify_crop(crop)
            lock_type(tr, t, conf)

        # Visualization
        draw_legend(frame)

        for tr in tracks:
            x1,y1,x2,y2 = tr.bbox
            t = tr.gate_type if tr.gate_type in TYPE_COLORS else "none"
            color = TYPE_COLORS.get(t, (200,200,200))

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            label = f"ID:{tr.track_id} {t} {tr.type_conf:.2f}"
            if tr.pending_type:
                label += f" -> {tr.pending_type}({tr.pending_hits})"

            cv2.putText(frame, label, (x1, max(15, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Tracks: {len(tracks)}",
                    (10, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2, cv2.LINE_AA)

        cv2.imshow("Vocab Gate Spotter (2-stage)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True,
                    help="Vocabulary folder with subfolders: square/circle/arch/flagpole")
    ap.add_argument("--video", default=None,
                    help="Optional MP4 path (if omitted uses camera)")
    args = ap.parse_args()
    run(args.vocab, args.video)
