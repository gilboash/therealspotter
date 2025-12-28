import os
import glob
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# SSL cert fix attempt (helps some mac installs)
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

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xA, yA = max(ax1,bx1), max(ay1,by1)
    xB, yB = min(ax2,bx2), min(ay2,by2)
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, ax2-ax1) * max(0, ay2-ay1)
    areaB = max(0, bx2-bx1) * max(0, by2-by1)
    denom = areaA + areaB - inter
    return float(inter) / float(denom) if denom > 0 else 0.0

def clamp_bbox(b: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = b
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return (x1,y1,x2,y2)

def crop_with_padding(frame: np.ndarray, bbox: Tuple[int,int,int,int], pad_frac: float = 0.10) -> np.ndarray:
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = bbox
    bw, bh = (x2-x1), (y2-y1)
    px, py = int(bw*pad_frac), int(bh*pad_frac)
    bb = clamp_bbox((x1-px, y1-py, x2+px, y2+py), w, h)
    x1,y1,x2,y2 = bb
    return frame[y1:y2, x1:x2].copy()


# ============================================================
# Stage 2: Vocabulary embedder
# ============================================================

class VocabEmbedder:
    def __init__(self, vocab_dir: str, device: str = "cpu", clip_model: str = "ViT-B-32", clip_pretrained: str = "openai"):
        self.vocab_dir = vocab_dir
        self.device = device
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained

        self.type_to_proto: Dict[str, np.ndarray] = {}
        self.available_types: List[str] = []

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

    def _iter_vocab_images(self, type_name: str) -> List[str]:
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
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
        hist = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
        hist = cv2.normalize(hist,None).flatten().astype(np.float32)
        n = np.linalg.norm(hist) + 1e-9
        return (hist / n).astype(np.float32)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        return self._embed_clip(bgr) if self.backend == "clip" else self._embed_hist(bgr)

    def _build_prototypes(self):
        types = [d for d in os.listdir(self.vocab_dir) if os.path.isdir(os.path.join(self.vocab_dir,d))]
        types = sorted(types)

        for t in types:
            imgs = self._iter_vocab_images(t)
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
            proto = np.mean(np.stack(feats,axis=0),axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            self.type_to_proto[t] = proto.astype(np.float32)
            self.available_types.append(t)

        if not self.available_types:
            raise RuntimeError(f"No vocab types found in {self.vocab_dir} (need subfolders per type)")

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float, Dict[str,float]]:
        feat = self.embed(crop_bgr)
        scores = {}
        best_t, best_s = "NONE", -1.0
        for t in self.available_types:
            s = float(np.dot(feat, self.type_to_proto[t]))
            scores[t] = s
            if s > best_s:
                best_s = s
                best_t = t
        return best_t, best_s, scores


# ============================================================
# Tracking + type locking
# ============================================================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int,int,int,int]
    last_seen: int
    locked_type: str = "NONE"
    locked_score: float = -1.0
    streak: int = 0
    miss: int = 0
    score_ema: float = 0.0  # smoothed type score
    det_ema: float = 0.0    # smoothed detector conf

class SimpleTracker:
    def __init__(self, iou_match: float, max_misses: int, lock_streak: int, lock_margin: float, ema_alpha: float):
        self.iou_match = iou_match
        self.max_misses = max_misses
        self.lock_streak = lock_streak
        self.lock_margin = lock_margin
        self.ema_alpha = ema_alpha

        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, frame_idx: int, dets: List[dict]) -> List[Track]:
        # age all tracks
        for t in self.tracks:
            t.miss += 1

        used = set()

        # match detections -> tracks by IoU
        for di, d in enumerate(dets):
            bb = d["bbox"]
            best_i, best_v = -1, 0.0
            for ti, t in enumerate(self.tracks):
                if ti in used:
                    continue
                v = iou(bb, t.bbox)
                if v > best_v:
                    best_v, best_i = v, ti

            if best_i >= 0 and best_v >= self.iou_match:
                t = self.tracks[best_i]
                used.add(best_i)

                t.bbox = bb
                t.last_seen = frame_idx
                t.miss = 0

                # smooth
                t.score_ema = (1-self.ema_alpha)*t.score_ema + self.ema_alpha*max(0.0, d["type_score"])
                t.det_ema   = (1-self.ema_alpha)*t.det_ema   + self.ema_alpha*max(0.0, d["det_conf"])

                self._lock_update(t, d["type"], d["type_score"])
            else:
                # new track
                t = Track(track_id=self.next_id, bbox=bb, last_seen=frame_idx)
                self.next_id += 1
                t.score_ema = max(0.0, d["type_score"])
                t.det_ema   = max(0.0, d["det_conf"])
                t.locked_type = d["type"] if d["type"] != "NONE" else "NONE"
                t.locked_score = float(d["type_score"])
                t.streak = 1 if t.locked_type != "NONE" else 0
                self.tracks.append(t)

        # prune stale
        self.tracks = [t for t in self.tracks if t.miss <= self.max_misses]
        return self.tracks

    def _lock_update(self, t: Track, new_type: str, new_score: float):
        new_score = float(new_score)

        if t.locked_type == "NONE":
            # acquire lock easily
            if new_type != "NONE":
                t.locked_type = new_type
                t.locked_score = new_score
                t.streak = 1
            return

        if new_type == t.locked_type:
            t.locked_score = max(t.locked_score, new_score)
            t.streak = min(t.streak + 1, 999)
            return

        # switch only if better by margin for lock_streak frames
        if new_type != "NONE" and new_score > (t.locked_score + self.lock_margin):
            t.streak += 1
            if t.streak >= self.lock_streak:
                t.locked_type = new_type
                t.locked_score = new_score
                t.streak = 1
        else:
            t.streak = 0


# ============================================================
# Visualization
# ============================================================

DEFAULT_COLORS = {
    "square": (255, 0, 0),      # blue
    "circle": (0, 255, 0),      # green
    "arch": (0, 255, 255),      # yellow
    "flagpole": (255, 0, 255),  # magenta
    "NONE": (128, 128, 128),    # gray
}

def pick_color(type_name: str) -> Tuple[int,int,int]:
    # best-effort mapping by substring, otherwise orange
    k = type_name.lower()
    if "square" in k: return DEFAULT_COLORS.get("square",(255,0,0))
    if "circle" in k: return DEFAULT_COLORS.get("circle",(0,255,0))
    if "arch" in k: return DEFAULT_COLORS.get("arch",(0,255,255))
    if "flag" in k: return DEFAULT_COLORS.get("flagpole",(255,0,255))
    if type_name == "NONE": return DEFAULT_COLORS["NONE"]
    return (0, 165, 255)

def draw_legend(frame: np.ndarray, types: List[str], hide_none: bool):
    x, y = 10, 20
    cv2.putText(frame, "Legend:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)
    y += 20
    for t in types:
        if hide_none and t == "NONE":
            continue
        c = pick_color(t)
        cv2.rectangle(frame, (x, y-12), (x+14, y+2), c, -1)
        cv2.putText(frame, t, (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
        y += 20

def draw_tracks(frame: np.ndarray, tracks: List[Track], hide_none: bool):
    for t in tracks:
        typ = t.locked_type or "NONE"
        if hide_none and typ == "NONE":
            continue
        x1,y1,x2,y2 = t.bbox
        c = pick_color(typ)
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 3)
        cv2.putText(frame, f"#{t.track_id} {typ} {t.score_ema:.2f}", (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Two-stage: permissive YOLO proposals + vocab embedding type match + tracking/lock")

    ap.add_argument("--mode", choices=["calib","learn","race"], default="learn")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--vocab", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")

    # show/hide NONE (default hidden)
    ap.add_argument("--hide-none", action="store_true", help="Hide NONE tracks (recommended)")

    # stage 1
    ap.add_argument("--det-model", type=str, default="yolov8n.pt")
    ap.add_argument("--det-conf", type=float, default=0.08)       # more permissive than 0.10
    ap.add_argument("--det-maxdet", type=int, default=60)

    # candidate limiting
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--min-type-score", type=float, default=0.15)  # used ONLY for final typing, not for ranking

    # tracker/lock
    ap.add_argument("--iou-match", type=float, default=0.25)
    ap.add_argument("--max-misses", type=int, default=10)
    ap.add_argument("--lock-streak", type=int, default=3)
    ap.add_argument("--lock-margin", type=float, default=0.06)
    ap.add_argument("--ema-alpha", type=float, default=0.35)

    args = ap.parse_args()

    embedder = VocabEmbedder(args.vocab, device=args.device)
    det = YOLO(args.det_model)

    tracker = SimpleTracker(
        iou_match=args.iou_match,
        max_misses=args.max_misses,
        lock_streak=args.lock_streak,
        lock_margin=args.lock_margin,
        ema_alpha=args.ema_alpha
    )

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open source")

    print(f"Embed backend: {embedder.backend} | Types: {embedder.available_types}")
    print("Press 'q' to quit.")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # Stage 1: proposals
        res = det(frame, conf=args.det_conf, verbose=False, max_det=args.det_maxdet)[0]
        proposals = []
        for b in res.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1,y1,x2,y2), W, H)
            proposals.append((bb, float(b.conf[0])))

        typed = []
        for bb, dconf in proposals:
            crop = crop_with_padding(frame, bb, pad_frac=0.12)
            best_t, best_s, _ = embedder.classify(crop)

            # IMPORTANT: do NOT collapse to NONE yet — keep score for ranking
            typed.append({
                "bbox": bb,
                "det_conf": float(dconf),
                "raw_type": best_t,
                "raw_type_score": float(best_s),
            })

        # Rank candidates by combined score:
        # - type similarity is primary
        # - det_conf helps break ties / keep plausible regions
        def combined_score(d):
            # normalize det_conf (~0..1) to small influence
            return float(d["raw_type_score"]) + 0.25 * float(d["det_conf"])

        typed.sort(key=combined_score, reverse=True)
        typed = typed[: max(1, args.max_candidates)]

        # Finalize type (apply min-type-score *after* topK)
        dets = []
        for d in typed:
            t = d["raw_type"]
            s = d["raw_type_score"]
            if s < args.min_type_score:
                t = "NONE"
            dets.append({
                "bbox": d["bbox"],
                "det_conf": d["det_conf"],
                "type": t,
                "type_score": s,
            })

        tracks = tracker.update(frame_idx, dets)

        # Visualization
        vis = frame.copy()

        # In calib/learn: also draw raw proposals lightly so you can debug "detector vs embedder"
        if args.mode in ("calib","learn"):
            for bb, dconf in proposals:
                x1,y1,x2,y2 = bb
                cv2.rectangle(vis, (x1,y1), (x2,y2), (70,70,70), 1)
                if args.mode == "calib":
                    cv2.putText(vis, f"{dconf:.2f}", (x1, max(15, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70,70,70), 1, cv2.LINE_AA)

        draw_tracks(vis, tracks, hide_none=args.hide_none)

        legend_types = list(embedder.available_types) + ["NONE"]
        draw_legend(vis, legend_types, hide_none=args.hide_none)

        cv2.putText(vis, f"MODE:{args.mode.upper()} proposals={len(proposals)} topK={len(dets)} tracks={len(tracks)} hideNONE={args.hide_none}",
                    (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)

        cv2.imshow("Vocab Gate Spotter", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
