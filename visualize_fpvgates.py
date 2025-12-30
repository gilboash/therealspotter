import os
import cv2
import glob

# --------------------------------------------------
# Update this if you add / reorder classes in YAML
# --------------------------------------------------
CLASS_NAMES = [
    "square",
    "arch",
    "circle",
    "flagpole",
]

COLORS = [
    (0, 255, 0),     # square - green
    (0, 255, 255),   # arch   - yellow
    (255, 0, 0),     # circle - blue
    (255, 0, 255),   # flag   - magenta
]


def load_yolo_labels(label_path, img_w, img_h):
    """
    Load YOLO-format labels and convert to pixel coordinates
    Returns: list of (class_id, x1, y1, x2, y2)
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)

            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            boxes.append((int(class_id), x1, y1, x2, y2))

    return boxes


def visualize_dataset(images_dir, labels_dir, max_images=50):
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    label_files = label_files[:max_images]

    if not label_files:
        print("No label files found.")
        return

    window_name = "FPV Gate Dataset Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for label_path in label_files:
        img_name = os.path.basename(label_path).replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] Image missing for {img_name}")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        boxes = load_yolo_labels(label_path, w, h)

        for class_id, x1, y1, x2, y2 in boxes:
            color = COLORS[class_id % len(COLORS)]
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"id{class_id}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"{class_name} ({class_id})",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, img)

        # 🔹 NEW: show filename in window title
        cv2.setWindowTitle(window_name, f"{window_name} — {img_name}")

        key = cv2.waitKey(0)
        if key == 27 or key == ord("q"):  # ESC or q
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize YOLO FPV gate labels")
    parser.add_argument("--images_dir", required=True, help="Path to images/train or images/val")
    parser.add_argument("--labels_dir", required=True, help="Path to labels/train or labels/val")
    parser.add_argument("--max_images", type=int, default=50)
    args = parser.parse_args()

    visualize_dataset(args.images_dir, args.labels_dir, args.max_images)
