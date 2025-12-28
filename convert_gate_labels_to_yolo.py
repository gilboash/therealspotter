# convert_gate_labels_to_yolo.py
import os
import argparse

def convert_label_file(src_path, dst_path):
    """
    Convert custom gate label format to YOLO detection format
    """
    yolo_lines = []

    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip malformed lines

            class_id = parts[0]
            cx = parts[1]
            cy = parts[2]
            w  = parts[3]
            h  = parts[4]

            yolo_lines.append(f"{class_id} {cx} {cy} {w} {h}")

    with open(dst_path, "w") as f:
        f.write("\n".join(yolo_lines))


def convert_folder(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.endswith(".txt"):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        convert_label_file(src_path, dst_path)

    print(f"✅ Converted labels written to: {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to original labels folder")
    parser.add_argument("--dst", required=True, help="Path to output YOLO labels folder")
    args = parser.parse_args()

    convert_folder(args.src, args.dst)
