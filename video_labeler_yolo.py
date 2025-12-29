import os
import cv2
import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def yolo_from_xyxy(x1, y1, x2, y2, w, h):
    # Ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return (xc / w, yc / h, bw / w, bh / h)

def xyxy_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def nice_frame_name(video_stem: str, frame_idx: int) -> str:
    return f"{video_stem}_{frame_idx:06d}"

# ----------------------------
# UI state
# ----------------------------

@dataclass
class Box:
    cls_id: int
    xyxy: Tuple[int, int, int, int]

class Annotator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.cur_cls = 0

        self.boxes: List[Box] = []
        self.drawing = False
        self.p1 = (0, 0)
        self.p2 = (0, 0)

        self.flash_msg = ""
        self.flash_until = 0.0

    def set_flash(self, msg: str, now: float, seconds: float = 1.2):
        self.flash_msg = msg
        self.flash_until = now + seconds

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.p1 = (x, y)
            self.p2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.p2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.p2 = (x, y)
            x1, y1, x2, y2 = xyxy_from_two_points(self.p1, self.p2)
            # ignore tiny boxes
            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                self.boxes.append(Box(cls_id=self.cur_cls, xyxy=(x1, y1, x2, y2)))

    def undo(self):
        if self.boxes:
            self.boxes.pop()

    def clear(self):
        self.boxes = []

    def draw_overlay(self, img, now: float):
        H, W = img.shape[:2]

        # draw existing boxes
        for b in self.boxes:
            x1, y1, x2, y2 = b.xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = self.class_names[b.cls_id] if 0 <= b.cls_id < len(self.class_names) else str(b.cls_id)
            cv2.putText(img, name, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # draw in-progress box
        if self.drawing:
            x1, y1, x2, y2 = xyxy_from_two_points(self.p1, self.p2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # HUD
        hud1 = f"Class: [{self.cur_cls}] {self.class_names[self.cur_cls]}   Boxes: {len(self.boxes)}"
        hud2 = "Keys: [a/d] prev/next  [j/k] -/+skip  [1..9] class  [u] undo  [c] clear  [s] save  [q] quit"
        cv2.putText(img, hud1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, hud2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

        if now < self.flash_until and self.flash_msg:
            cv2.putText(img, self.flash_msg, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Video -> YOLO annotator (draw boxes, set class, save YOLO labels)")
    ap.add_argument("--video", required=True, help="Path to .mp4")
    ap.add_argument("--out", required=True, help="Output dataset root (will create images/{split}, labels/{split})")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Which split to write into")
    ap.add_argument("--classes", default="square,circle,arch,flagpole", help="Comma-separated class names")
    ap.add_argument("--start", type=int, default=0, help="Start frame index")
    ap.add_argument("--step", type=int, default=1, help="Frame step when moving next/prev (default 1)")
    ap.add_argument("--resize", type=str, default=None, help="Optional resize WxH, e.g. 1280x720 (keeps labels correct)")
    ap.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality 0-100")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        raise SystemExit("No classes provided. Use --classes 'gate' or similar.")

    # Prepare output dirs
    img_dir = os.path.join(args.out, "images", args.split)
    lab_dir = os.path.join(args.out, "labels", args.split)
    ensure_dir(img_dir)
    ensure_dir(lab_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # Resize parsing
    resize_wh = None
    if args.resize:
        try:
            w_str, h_str = args.resize.lower().split("x")
            resize_wh = (int(w_str), int(h_str))
        except Exception:
            raise SystemExit("Bad --resize. Use like 1280x720")

    ann = Annotator(class_names=class_names)

    win = "YOLO Video Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, ann.on_mouse)

    def read_frame(idx: int):
        idx = max(0, idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        if resize_wh:
            frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
        return frame

    frame_idx = args.start
    frame = read_frame(frame_idx)
    if frame is None:
        raise SystemExit(f"Could not read frame {frame_idx} from video.")

    skip = max(1, args.step)

    while True:
        now = cv2.getTickCount() / cv2.getTickFrequency()
        vis = frame.copy()
        ann.draw_overlay(vis, now)

        # bottom-right frame counter
        H, W = vis.shape[:2]
        count_text = f"Frame {frame_idx}" + (f"/{total_frames-1}" if total_frames > 0 else "")
        cv2.putText(vis, count_text, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            break

        # class selection 1..9
        if ord("1") <= key <= ord("9"):
            cls = (key - ord("1"))
            if cls < len(class_names):
                ann.cur_cls = cls

        if key == ord("u"):
            ann.undo()

        if key == ord("c"):
            ann.clear()

        # prev/next
        if key == ord("d"):
            frame_idx += skip
            ann.clear()
            frame = read_frame(frame_idx)
            if frame is None:
                break

        if key == ord("a"):
            frame_idx = max(0, frame_idx - skip)
            ann.clear()
            frame = read_frame(frame_idx)
            if frame is None:
                break

        # adjust skip with j/k
        if key == ord("k"):
            skip = min(500, skip + 1)
            ann.set_flash(f"skip = {skip}", now, 0.8)

        if key == ord("j"):
            skip = max(1, skip - 1)
            ann.set_flash(f"skip = {skip}", now, 0.8)

        # save
        if key == ord("s"):
            if not ann.boxes:
                ann.set_flash("No boxes to save", now, 1.0)
                continue

            name = nice_frame_name(video_stem, frame_idx)
            out_img = os.path.join(img_dir, f"{name}.jpg")
            out_lab = os.path.join(lab_dir, f"{name}.txt")

            Hf, Wf = frame.shape[:2]

            # write label file (YOLO format)
            lines = []
            for b in ann.boxes:
                x1, y1, x2, y2 = b.xyxy
                x1 = clamp(x1, 0, Wf - 1)
                x2 = clamp(x2, 0, Wf - 1)
                y1 = clamp(y1, 0, Hf - 1)
                y2 = clamp(y2, 0, Hf - 1)
                xc, yc, bw, bh = yolo_from_xyxy(x1, y1, x2, y2, Wf, Hf)
                lines.append(f"{b.cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            with open(out_lab, "w") as f:
                f.write("\n".join(lines) + "\n")

            # write image
            cv2.imwrite(out_img, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])

            ann.set_flash(f"SAVED: {os.path.basename(out_img)} (+labels)", now, 1.3)

            # auto-advance to next frame after save (common workflow)
            frame_idx += skip
            ann.clear()
            frame = read_frame(frame_idx)
            if frame is None:
                break

    cap.release()
    cv2.destroyAllWindows()

    # also write classes file (helpful for reference)
    classes_txt = os.path.join(args.out, "classes.txt")
    try:
        with open(classes_txt, "w") as f:
            f.write("\n".join(class_names) + "\n")
    except Exception:
        pass

    print("Done.")

if __name__ == "__main__":
    main()
