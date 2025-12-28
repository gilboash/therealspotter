# visualize_fpvgates_full.py
import os
import cv2
import glob
from detector import GateDetector

def load_yolo_labels(label_path, img_w, img_h):
    """
    Load YOLO-format labels and convert to pixel coordinates
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            x1 = int((x_center - width/2) * img_w)
            y1 = int((y_center - height/2) * img_h)
            x2 = int((x_center + width/2) * img_w)
            y2 = int((y_center + height/2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes

def visualize_dataset(images_dir, labels_dir, max_images=50, model_path=None, conf=0.65):
    """
    Visualize images with bounding boxes from labels and optional model predictions
    """
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    label_files = label_files[:max_images]

    # load trained YOLO model if provided
    if model_path:
        detector = GateDetector(model_path=model_path, conf=conf)
    else:
        detector = None

    for label_path in label_files:
        img_name = os.path.basename(label_path).replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Image missing for {img_name}")
            continue

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # draw label boxes
        boxes = load_yolo_labels(label_path, img_w, img_h)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green = label

        # draw model predictions if detector provided
        if detector:
            detections = detector.detect(img)
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = prediction

        cv2.imshow("Gate Visualization", img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to images/train or images/val")
    parser.add_argument("--labels_dir", required=True, help="Path to labels/train or labels/val")
    parser.add_argument("--max_images", type=int, default=50, help="Maximum images to visualize")
    parser.add_argument("--model_path", type=str, default=None, help="Optional trained YOLO model path")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold for model predictions")
    args = parser.parse_args()

    visualize_dataset(args.images_dir, args.labels_dir, args.max_images, args.model_path, args.conf)
