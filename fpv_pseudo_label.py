# fpv_pseudo_label_multibox.py
#
# Option A: Save MULTIPLE boxes per frame (YOLO multi-line label files)
# Adapted to your current implementation (Ultralytics YOLO directly; no detector/crossing/track modules).
#
# Notes:
# - Writes one image per selected frame, and a .txt with 1..N lines (each a bbox).
# - Supports single-class (default) OR multi-class (use YOLO class id).
# - Includes simple anti-dup filtering per frame, and optional cooldown between saved frames.
#
# Usage example:
#   python fpv_pseudo_label_multibox.py \
#     --video_dir ./videos \
#     --out_dir ./dataset \
#     --yolo_model yolov8n.pt \
#     --conf 0.25 \
#     --maxdet 50 \
#     --single_class \
#     --min_area_ratio 0.002 \
#     --save_every_sec 0.0 \
#     --max_labels_per_frame 10
#
import os
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# FS helpers
# -----------------------------
def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Geometry helpers
# -----------------------------
def clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


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


def bbox_area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def bbox_aspect(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return float(w) / float(h)


# -----------------------------
# YOLO label writer (multi-line)
# -----------------------------
def yolo_line_from_bbox(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    class_id: int,
) -> str:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / float(img_w)
    y_center = ((y1 + y2) / 2.0) / float(img_h)
    width = (x2 - x1) / float(img_w)
    height = (y2 - y1) / float(img_h)
    return f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# -----------------------------
# Filtering / selection
# -----------------------------
def filter_dets(
    dets: List[dict],
    img_w: int,
    img_h: int,
    *,
    min_area_ratio: float,
    aspect_min: float,
    aspect_max: float,
) -> List[dict]:
    out = []
    frame_area = float(max(1, img_w * img_h))
    for d in dets:
        bb = d["bbox"]
        ar = bbox_area(bb) / frame_area
        asp = bbox_aspect(bb)
        if ar < float(min_area_ratio):
            continue
        if asp < float(aspect_min) or asp > float(aspect_max):
            continue
        out.append(d)
    return out


def nms_like_dedupe(
    dets: List[dict],
    *,
    iou_thresh: float,
    max_keep: int,
) -> List[dict]:
    """
    Simple greedy de-dup by IOU, assumes dets pre-sorted by score descending.
    Keeps up to max_keep.
    """
    kept: List[dict] = []
    for d in dets:
        bb = d["bbox"]
        ok = True
        for k in kept:
            if iou(bb, k["bbox"]) >= float(iou_thresh):
                ok = False
                break
        if ok:
            kept.append(d)
        if len(kept) >= int(max_keep):
            break
    return kept


# -----------------------------
# Core processing
# -----------------------------
def detect_with_ultralytics(model: YOLO, frame: np.ndarray, conf: float, maxdet: int) -> List[dict]:
    """
    Returns list of dicts: {bbox, conf, cls}
    bbox is int xyxy clamped.
    """
    H, W = frame.shape[:2]
    r = model(frame, conf=float(conf), verbose=False, max_det=int(maxdet))[0]

    dets: List[dict] = []
    if r.boxes is None:
        return dets

    for b in r.boxes:
        if b.xyxy is None:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        bb = clamp_bbox((int(x1), int(y1), int(x2), int(y2)), W, H)

        c = float(b.conf[0]) if b.conf is not None else 0.0
        cls_id = int(b.cls[0]) if b.cls is not None else 0

        dets.append({"bbox": bb, "conf": c, "cls": cls_id})

    dets.sort(key=lambda d: float(d["conf"]), reverse=True)
    return dets


def process_video(
    video_path: str,
    out_image_dir: str,
    out_label_dir: str,
    model: YOLO,
    *,
    conf: float,
    maxdet: int,
    single_class: bool,
    min_area_ratio: float,
    aspect_min: float,
    aspect_max: float,
    max_labels_per_frame: int,
    per_frame_iou_dedupe: float,
    save_every_sec: float,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved_count = 0
    last_saved_t = -1e9

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < int(start_frame):
            frame_idx += 1
            continue
        if end_frame is not None and frame_idx > int(end_frame):
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        now_sec = float(pos_msec) / 1000.0

        # optional cooldown between saves (prevents huge near-duplicate datasets)
        if float(save_every_sec) > 0.0 and (now_sec - last_saved_t) < float(save_every_sec):
            frame_idx += 1
            continue

        img_h, img_w = frame.shape[:2]

        dets = detect_with_ultralytics(model, frame, conf=conf, maxdet=maxdet)
        dets = filter_dets(
            dets,
            img_w=img_w,
            img_h=img_h,
            min_area_ratio=min_area_ratio,
            aspect_min=aspect_min,
            aspect_max=aspect_max,
        )

        if not dets:
            frame_idx += 1
            continue

        # De-dup overlapped detections and cap per-frame labels
        dets = nms_like_dedupe(
            dets,
            iou_thresh=per_frame_iou_dedupe,
            max_keep=max_labels_per_frame,
        )

        if not dets:
            frame_idx += 1
            continue

        # Save image once
        img_filename = f"{frame_idx:06d}.jpg"
        txt_filename = f"{frame_idx:06d}.txt"
        img_path = os.path.join(out_image_dir, img_filename)
        txt_path = os.path.join(out_label_dir, txt_filename)

        cv2.imwrite(img_path, frame)

        # Write multiple lines (YOLO multi-object labels)
        lines = []
        for d in dets:
            cls_out = 0 if single_class else int(d["cls"])
            lines.append(yolo_line_from_bbox(d["bbox"], img_w, img_h, cls_out))

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        saved_count += 1
        last_saved_t = now_sec

        frame_idx += 1

    cap.release()
    return saved_count


# -----------------------------
# Train/val split by videos
# -----------------------------
def list_videos(video_dir: str) -> List[str]:
    vids = []
    for fn in os.listdir(video_dir):
        if fn.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
            vids.append(os.path.join(video_dir, fn))
    vids.sort()
    return vids


def main():
    parser = argparse.ArgumentParser(description="FPV pseudo-label dataset generator (multi-box per frame, YOLO format)")

    parser.add_argument("--video_dir", required=True, help="Folder with videos")
    parser.add_argument("--out_dir", required=True, help="Output dataset folder")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="Ultralytics YOLO model path (.pt)")

    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--maxdet", type=int, default=50, help="YOLO max detections per frame")

    parser.add_argument("--single_class", action="store_true", help="Write all labels as class 0 (single class dataset)")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split fraction by videos (sorted order)")

    # geometry filters (per detection)
    parser.add_argument("--min_area_ratio", type=float, default=0.002, help="Min bbox area / frame area")
    parser.add_argument("--aspect_min", type=float, default=0.3)
    parser.add_argument("--aspect_max", type=float, default=3.0)

    # multi-box controls
    parser.add_argument("--max_labels_per_frame", type=int, default=10, help="Max boxes to write in one frame")
    parser.add_argument("--per_frame_iou_dedupe", type=float, default=0.6, help="De-dup overlapped boxes within a frame")

    # saving cadence
    parser.add_argument("--save_every_sec", type=float, default=0.0, help="Cooldown between saved frames (0 disables)")

    # optional frame range
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)

    args = parser.parse_args()

    # Setup output folders
    images_train = os.path.join(args.out_dir, "images", "train")
    labels_train = os.path.join(args.out_dir, "labels", "train")
    images_val = os.path.join(args.out_dir, "images", "val")
    labels_val = os.path.join(args.out_dir, "labels", "val")

    mkdir(images_train)
    mkdir(labels_train)
    mkdir(images_val)
    mkdir(labels_val)

    videos = list_videos(args.video_dir)
    if not videos:
        raise RuntimeError(f"No videos found in {args.video_dir}")

    n_train = int(len(videos) * float(args.train_split))

    model = YOLO(args.yolo_model)

    total_saved = 0
    for i, vid in enumerate(videos):
        out_img_dir = images_train if i < n_train else images_val
        out_lbl_dir = labels_train if i < n_train else labels_val

        print(f"Processing {vid} → {out_img_dir}")
        saved = process_video(
            video_path=vid,
            out_image_dir=out_img_dir,
            out_label_dir=out_lbl_dir,
            model=model,
            conf=float(args.conf),
            maxdet=int(args.maxdet),
            single_class=bool(args.single_class),
            min_area_ratio=float(args.min_area_ratio),
            aspect_min=float(args.aspect_min),
            aspect_max=float(args.aspect_max),
            max_labels_per_frame=int(args.max_labels_per_frame),
            per_frame_iou_dedupe=float(args.per_frame_iou_dedupe),
            save_every_sec=float(args.save_every_sec),
            start_frame=int(args.start_frame),
            end_frame=(None if args.end_frame is None else int(args.end_frame)),
        )
        total_saved += saved
        print(f"  saved frames: {saved}")

    print(f"Pseudo-label dataset generation complete! total saved frames: {total_saved}")


if __name__ == "__main__":
    main()
