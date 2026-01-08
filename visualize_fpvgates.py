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


def _draw_help_overlay(img):
    """
    Draw a small help overlay with hotkeys.
    """
    overlay_h = 70
    x1, y1 = 0, 0
    x2, y2 = img.shape[1], min(img.shape[0], overlay_h)

    # dark translucent bar
    bar = img[y1:y2, x1:x2].copy()
    dark = (bar * 0.35).astype(bar.dtype)
    img[y1:y2, x1:x2] = dark

    lines = [
        "Keys: [Enter/Space/Right] next   [q/ESC] quit   [d] delete image+label",
        "Delete asks for confirmation: press 'y' to confirm, anything else cancels.",
    ]
    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24


def visualize_dataset(images_dir, labels_dir, max_images=50):
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    label_files = label_files[:max_images]

    if not label_files:
        print("No label files found.")
        return

    window_name = "FPV Gate Dataset Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(label_files):
        label_path = label_files[i]
        img_name = os.path.basename(label_path).replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] Image missing for {img_name}")
            i += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed reading {img_path}")
            i += 1
            continue

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

        # help overlay
        _draw_help_overlay(img)

        cv2.imshow(window_name, img)
        cv2.setWindowTitle(window_name, f"{window_name} — {img_name} ({i+1}/{len(label_files)})")

        key = cv2.waitKey(0) & 0xFF

        # quit
        if key == 27 or key == ord("q"):  # ESC or q
            break

        # delete current image+label
        if key == ord("d"):
            msg = f"DELETE {img_name} and {os.path.basename(label_path)} ? (y/n)"
            print("[DELETE?]", msg)

            # show a quick confirmation overlay
            confirm = img.copy()
            cv2.putText(confirm, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow(window_name, confirm)
            ckey = cv2.waitKey(0) & 0xFF

            if ckey in (ord("y"), ord("Y")):
                ok_img = True
                ok_lbl = True

                try:
                    os.remove(img_path)
                except Exception as e:
                    ok_img = False
                    print(f"[ERR] Failed deleting image: {img_path} ({e})")

                try:
                    os.remove(label_path)
                except Exception as e:
                    ok_lbl = False
                    print(f"[ERR] Failed deleting label: {label_path} ({e})")

                if ok_img and ok_lbl:
                    print(f"[OK] Deleted: {img_name} + {os.path.basename(label_path)}")
                else:
                    print(f"[WARN] Partial delete. image_ok={ok_img} label_ok={ok_lbl}")

                # remove from list so viewer continues correctly
                label_files.pop(i)
                # don't increment i; next item shifts into i
                if not label_files:
                    print("No more files.")
                    break
                continue
            else:
                print("[CANCEL] delete")
                # stay on same image unless user presses next
                continue

        # next (Enter/Space/right arrow)
        if key in (13, 32, 83):  # Enter, Space, Right arrow
            i += 1
            continue

        # default: next
        i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize YOLO FPV gate labels")
    parser.add_argument("--images_dir", required=True, help="Path to images/train or images/val")
    parser.add_argument("--labels_dir", required=True, help="Path to labels/train or labels/val")
    parser.add_argument("--max_images", type=int, default=50)
    args = parser.parse_args()

    visualize_dataset(args.images_dir, args.labels_dir, args.max_images)
