import os
import cv2
import glob
import numpy as np
import argparse
from ultralytics import YOLO

# ----------------------------
# Utilities
# ----------------------------

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def crop_with_pad(img, bbox, pad=0.08):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)
    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(w-1, x2 + px); y2 = min(h-1, y2 + py)
    return img[y1:y2, x1:x2]

# ----------------------------
# Embedding backends
# ----------------------------

class Embedder:
    """
    Two options:
    1) 'opencv': tiny, no internet, weaker but works for "shape-ish" categories if exemplars are good.
    2) 'clip': best, but requires installing transformers/open_clip and downloading weights (needs SSL fixed).
    """
    def __init__(self, backend="opencv"):
        self.backend = backend
        if backend == "opencv":
            # Simple global descriptor: HOG + color histogram (no downloads).
            # Surprisingly decent for gate-shape types when vocab exemplars are tight crops.
            pass
        elif backend == "clip":
            # Lazy import so opencv backend doesn't require it.
            import torch
            import open_clip
            self.torch = torch
            self.open_clip = open_clip
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.model = self.model.to(self.device).eval()
        else:
            raise ValueError("backend must be 'opencv' or 'clip'")

    def embed(self, bgr_img: np.ndarray) -> np.ndarray:
        if bgr_img is None or bgr_img.size == 0:
            return None

        if self.backend == "opencv":
            # HOG + HSV histogram
            img = cv2.resize(bgr_img, (128, 128))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hog = cv2.HOGDescriptor(
                _winSize=(128,128),
                _blockSize=(32,32),
                _blockStride=(16,16),
                _cellSize=(16,16),
                _nbins=9
            )
            hog_feat = hog.compute(gray).flatten()

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1], None, [24,24], [0,180,0,256]).flatten()
            hist = hist / (np.sum(hist) + 1e-12)

            feat = np.concatenate([hog_feat, hist]).astype(np.float32)
            return l2norm(feat)

        elif self.backend == "clip":
            import PIL.Image
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            pil = PIL.Image.fromarray(rgb)
            image = self.preprocess(pil).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                feat = self.model.encode_image(image).float().cpu().numpy().flatten()
            return l2norm(feat)

# ----------------------------
# Vocabulary prototypes
# ----------------------------

def load_vocab(vocab_dir: str, embedder: Embedder):
    """
    vocab_dir/
      square/*.jpg
      arch/*.jpg
      circle/*.jpg
      flagpole/*.jpg
    """
    prototypes = {}
    for gate_type in sorted(os.listdir(vocab_dir)):
        tdir = os.path.join(vocab_dir, gate_type)
        if not os.path.isdir(tdir):
            continue
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png"):
            imgs += glob.glob(os.path.join(tdir, ext))
        if not imgs:
            continue

        feats = []
        for p in imgs:
            im = cv2.imread(p)
            if im is None:
                continue
            f = embedder.embed(im)
            if f is not None:
                feats.append(f)

        if feats:
            proto = l2norm(np.mean(np.stack(feats, axis=0), axis=0))
            prototypes[gate_type] = proto
            print(f"Loaded vocab type '{gate_type}' with {len(feats)} examples")
    if not prototypes:
        raise RuntimeError("No vocabulary images found. Check vocab folder structure.")
    return prototypes

# ----------------------------
# Simple IoU tracker (no dependencies)
# ----------------------------

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xA = max(ax1, bx1); yA = max(ay1, by1)
    xB = min(ax2, bx2); yB = min(ay2, by2)
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter <= 0:
        return 0.0
    areaA = (ax2-ax1)*(ay2-ay1)
    areaB = (bx2-bx1)*(by2-by1)
    return inter / float(areaA + areaB - inter + 1e-12)

class TrackManager:
    def __init__(self, iou_match=0.3, ttl=10):
        self.iou_match = iou_match
        self.ttl = ttl
        self.next_id = 1
        self.tracks = {}  # id -> dict(bbox, age, type, score)

    def update(self, detections):
        # age existing
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.ttl:
                del self.tracks[tid]

        # match detections
        for det in detections:
            bbox = det["bbox"]
            best_id = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                v = iou(bbox, tr["bbox"])
                if v > best_iou:
                    best_iou = v
                    best_id = tid
            if best_id is not None and best_iou >= self.iou_match:
                self.tracks[best_id]["bbox"] = bbox
                self.tracks[best_id]["age"] = 0
                self.tracks[best_id]["det_conf"] = det.get("det_conf", 0.0)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = dict(
                    bbox=bbox, age=0, det_conf=det.get("det_conf", 0.0),
                    gate_type="none", match_score=0.0
                )
        return self.tracks

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="mp4 file or webcam if omitted")
    ap.add_argument("--vocab", type=str, required=True, help="vocab directory with subfolders per type")
    ap.add_argument("--embed", type=str, default="opencv", choices=["opencv","clip"], help="embedding backend")
    ap.add_argument("--yolo", type=str, default="yolov8n.pt", help="proposal model (general)")
    ap.add_argument("--conf", type=float, default=0.15, help="proposal confidence (permissive)")
    ap.add_argument("--match_thr", type=float, default=0.25, help="cosine similarity threshold for type match")
    args = ap.parse_args()

    embedder = Embedder(args.embed)
    prototypes = load_vocab(args.vocab, embedder)

    model = YOLO(args.yolo)  # general, permissive proposals
    tracker = TrackManager(iou_match=0.3, ttl=12)

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- Stage 1: permissive proposals (all classes, treat as objectness proposals)
        res = model(frame, conf=args.conf, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            dets.append({"bbox": (x1,y1,x2,y2), "det_conf": float(b.conf[0])})

        tracks = tracker.update(dets)

        # ---- Stage 2: vocab matching per track
        for tid, tr in tracks.items():
            crop = crop_with_pad(frame, tr["bbox"], pad=0.10)
            feat = embedder.embed(crop)
            if feat is None:
                tr["gate_type"] = "none"
                tr["match_score"] = 0.0
                continue

            best_t = "none"
            best_s = -1.0
            for t, proto in prototypes.items():
                s = cosine(feat, proto)
                if s > best_s:
                    best_s = s
                    best_t = t

            if best_s >= args.match_thr:
                tr["gate_type"] = best_t
                tr["match_score"] = best_s
            else:
                tr["gate_type"] = "none"
                tr["match_score"] = best_s

        # ---- Visualization
        for tid, tr in tracks.items():
            x1,y1,x2,y2 = tr["bbox"]
            t = tr["gate_type"]
            s = tr["match_score"]
            dc = tr.get("det_conf", 0.0)

            color = (0,255,0) if t != "none" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"#{tid} {t} sim={s:.2f} det={dc:.2f}",
                        (x1, max(y1-8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"embed={args.embed}  match_thr={args.match_thr}  conf={args.conf}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Vocab Spotter (green=matched, red=none)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
