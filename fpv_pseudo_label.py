# fpv_pseudo_label_permissive.py
import os
import cv2
import numpy as np
from detector import GateDetector
from crossing import CrossingDetector
from track import Track

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_yolo_label(txt_path, bbox, img_w, img_h):
    """
    Converts bbox (x1,y1,x2,y2) to YOLO format and saves to txt file.
    Assumes single class (class_id = 0)
    """
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    with open(txt_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_video(video_path, out_image_dir, out_label_dir, permissive_detector):
    cap = cv2.VideoCapture(video_path)
    crossing = CrossingDetector()
    track = Track(expected_gates=50)  # high number to capture all gates

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]

        # Use permissive detector
        detections = permissive_detector.detect(frame)

        for d in detections:
            # Use crossing logic to approximate gate frames
            if crossing.check_crossing(d["bbox"]):
                track.add_gate(d, np.array([0, 0, 1]))
                img_filename = f"{frame_idx:06d}.jpg"
                txt_filename = f"{frame_idx:06d}.txt"
                cv2.imwrite(os.path.join(out_image_dir, img_filename), frame)
                save_yolo_label(os.path.join(out_label_dir, txt_filename), d["bbox"], img_w, img_h)

        frame_idx += 1

    cap.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Folder with MP4 videos")
    parser.add_argument("--out_dir", required=True, help="Output dataset folder")
    args = parser.parse_args()

    # Setup output folders
    images_train = os.path.join(args.out_dir, "images", "train")
    labels_train = os.path.join(args.out_dir, "labels", "train")
    mkdir(images_train)
    mkdir(labels_train)

    images_val = os.path.join(args.out_dir, "images", "val")
    labels_val = os.path.join(args.out_dir, "labels", "val")
    mkdir(images_val)
    mkdir(labels_val)

    # Create a permissive detector just for pseudo-labeling
    permissive_detector = GateDetector(
        model_path="yolov8n.pt",
        conf=0.3,            # lower threshold
        min_area_ratio=0.002, # allow smaller boxes
        aspect_min=0.3,
        aspect_max=3.0,
        persist_frames=1      # no temporal filtering
    )

    # Simple train/val split by videos
    videos = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]
    n_train = int(len(videos) * 0.8)
    for i, vid in enumerate(videos):
        out_img_dir = images_train if i < n_train else images_val
        out_lbl_dir = labels_train if i < n_train else labels_val
        print(f"Processing {vid} → {out_img_dir}")
        process_video(vid, out_img_dir, out_lbl_dir, permissive_detector)

    print("Pseudo-label dataset generation complete!")
