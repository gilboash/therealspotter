import os, glob

DATA = "./datastuff/fpv_dataset"   # <-- adjust if needed
splits = ["train", "val"]
nc = 4  # number of classes in your YAML

def bad(msg, p): 
    print(f"[BAD] {msg}: {p}")

for sp in splits:
    img_dir = os.path.join(DATA, "images", sp)
    lab_dir = os.path.join(DATA, "labels", sp)

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs += glob.glob(os.path.join(img_dir, ext))

    imgs = sorted(imgs)

    print(f"\n================ {sp.upper()} ================")

    if not imgs:
        print(f"[WARN] no images in {img_dir}")
        continue

    missing_labels = []
    empty_labels = []
    invalid_lines = []

    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lab = os.path.join(lab_dir, stem + ".txt")

        if not os.path.exists(lab):
            missing_labels.append(lab)
            continue

        txt = open(lab, "r").read().strip()
        if not txt:
            empty_labels.append(lab)
            continue

        for i, line in enumerate(txt.splitlines()):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_lines.append((lab, i, "wrong column count"))
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except Exception:
                invalid_lines.append((lab, i, "non-numeric values"))
                continue

            if not (0 <= cls < nc):
                invalid_lines.append((lab, i, f"class id out of range ({cls})"))

            if not (
                0.0 <= x <= 1.0 and
                0.0 <= y <= 1.0 and
                0.0 < w <= 1.0 and
                0.0 < h <= 1.0
            ):
                invalid_lines.append((lab, i, "coords out of range"))

    print(f"Images total        : {len(imgs)}")
    print(f"Missing labels      : {len(missing_labels)}")
    print(f"Empty label files   : {len(empty_labels)}")
    print(f"Invalid label lines : {len(invalid_lines)}")

    if missing_labels:
        print("\n--- Missing label files ---")
        for p in missing_labels:
            print(" ", p)

    if empty_labels:
        print("\n--- EMPTY label files (0 boxes) ---")
        for p in empty_labels:
            print(" ", p)

    if invalid_lines:
        print("\n--- Invalid label lines ---")
        for lab, line_idx, reason in invalid_lines:
            print(f" {lab}  line {line_idx}: {reason}")
