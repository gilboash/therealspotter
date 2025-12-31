#!/usr/bin/env python3
"""
pipeline_sanity_train_eval.py

Sanity check -> (optional) train -> evaluate on test

Expected dataset layout:
  DATA/
    images/
      train/
      val/
      test/     (optional but recommended for evaluation)
    labels/
      train/
      val/
      test/

Your data YAML should point to DATA and include train/val (and optionally test).
Example fpv_gate.yaml:
  path: ./datastuff/fpv_dataset
  train: images/train
  val: images/val
  test: images/test         # (recommended)
  nc: 4
  names: ['square','arch','circle','flagpole']
"""

import os
import sys
import glob
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ultralytics must be installed in the venv you run this with
from ultralytics import YOLO


# ----------------------------
# Sanity checking
# ----------------------------

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

@dataclass
class InvalidLine:
    lab_path: str
    line_idx: int
    reason: str
    line: str

def _bad(msg: str, p: str):
    print(f"[BAD] {msg}: {p}")

def list_images(img_dir: str) -> List[str]:
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(imgs)

def sanity_split(data_root: str, split: str, nc: int, print_paths: bool = True) -> Tuple[int, int, int, int]:
    """
    Returns:
      (num_images, num_missing_labels, num_empty_labels, num_invalid_lines)
    """
    img_dir = os.path.join(data_root, "images", split)
    lab_dir = os.path.join(data_root, "labels", split)

    print(f"\n================ {split.upper()} ================")
    if not os.path.isdir(img_dir):
        print(f"[WARN] missing images dir: {img_dir}")
        return (0, 0, 0, 0)
    if not os.path.isdir(lab_dir):
        print(f"[WARN] missing labels dir: {lab_dir}")
        return (0, 0, 0, 0)

    imgs = list_images(img_dir)
    if not imgs:
        print(f"[WARN] no images found in {img_dir}")
        return (0, 0, 0, 0)

    missing_labels: List[str] = []
    empty_labels: List[str] = []
    invalid_lines: List[InvalidLine] = []

    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lab = os.path.join(lab_dir, stem + ".txt")

        if not os.path.exists(lab):
            missing_labels.append(lab)
            continue

        txt = open(lab, "r", encoding="utf-8", errors="ignore").read().strip()
        if not txt:
            empty_labels.append(lab)
            continue

        for i, line in enumerate(txt.splitlines()):
            raw = line
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_lines.append(InvalidLine(lab, i, "wrong column count (!= 5)", raw))
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except Exception:
                invalid_lines.append(InvalidLine(lab, i, "non-numeric values", raw))
                continue

            if not (0 <= cls < nc):
                invalid_lines.append(InvalidLine(lab, i, f"class id out of range ({cls})", raw))

            # YOLO expects normalized center/wh
            if not (
                0.0 <= x <= 1.0 and
                0.0 <= y <= 1.0 and
                0.0 <  w <= 1.0 and
                0.0 <  h <= 1.0
            ):
                invalid_lines.append(InvalidLine(lab, i, "coords out of range", raw))

    print(f"Images total        : {len(imgs)}")
    print(f"Missing labels      : {len(missing_labels)}")
    print(f"Empty label files   : {len(empty_labels)}")
    print(f"Invalid label lines : {len(invalid_lines)}")

    if print_paths:
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
            for it in invalid_lines[:200]:  # cap spam
                print(f" {it.lab_path}  line {it.line_idx}: {it.reason}\n    {it.line}")
            if len(invalid_lines) > 200:
                print(f" ... ({len(invalid_lines)-200} more)")

    return (len(imgs), len(missing_labels), len(empty_labels), len(invalid_lines))


# ----------------------------
# Train + Eval
# ----------------------------

def run_train(
    data_yaml: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
) -> str:
    """
    Trains and returns best weights path.
    """
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        project=project,
        name=name,
    )

    # Ultralytics convention: best.pt is in runs/<task>/<name>/weights/best.pt
    # results.save_dir is a Path-like
    save_dir = str(getattr(results, "save_dir", "")) if results is not None else ""
    if not save_dir:
        # fallback guess
        save_dir = os.path.join(project, name)

    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        # sometimes it's "best.pt" elsewhere; try common fallbacks
        alt = glob.glob(os.path.join(save_dir, "**", "best.pt"), recursive=True)
        if alt:
            best_pt = alt[0]

    if not os.path.exists(best_pt):
        raise RuntimeError(f"Training finished but best.pt not found under: {save_dir}")

    return best_pt


def run_eval(weights: str, data_yaml: str, split: str, imgsz: int, batch: int, device: str):
    """
    Evaluates and prints precision/recall + mAP.
    """
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split=split,   # 'val' or 'test' (if YAML has test)
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
    )

    # Ultralytics metrics object structure:
    # metrics.box.p, metrics.box.r, metrics.box.map, metrics.box.map50, metrics.box.map75
    box = getattr(metrics, "box", None)
    if box is None:
        print("[WARN] Could not find metrics.box in returned metrics object.")
        print(metrics)
        return

    p = getattr(box, "p", None)
    r = getattr(box, "r", None)
    map50 = getattr(box, "map50", None)
    map5095 = getattr(box, "map", None)

    print("\n================ EVAL SUMMARY ================")
    print(f"Weights: {weights}")
    print(f"Split  : {split}")
    if map5095 is not None:
        print(f"mAP@50:95: {float(map5095):.4f}")
    if map50 is not None:
        print(f"mAP@50    : {float(map50):.4f}")

    # Per-class P/R if available (usually numpy arrays)
    if p is not None and r is not None:
        try:
            import numpy as np  # local import
            p_arr = np.array(p).astype(float).reshape(-1)
            r_arr = np.array(r).astype(float).reshape(-1)
            print("\nPer-class Precision / Recall:")
            for i in range(min(len(p_arr), len(r_arr))):
                print(f"  class {i}: P={p_arr[i]:.3f}  R={r_arr[i]:.3f}")
        except Exception:
            # fallback print
            print(f"Precision: {p}")
            print(f"Recall   : {r}")


def main():
    ap = argparse.ArgumentParser(description="Sanity check -> (optional) train -> eval (precision/recall)")

    ap.add_argument("--data-root", required=True, help="Dataset root (contains images/ and labels/)")
    ap.add_argument("--data-yaml", required=True, help="YOLO data YAML path (train/val[/test], nc, names)")
    ap.add_argument("--nc", type=int, required=True, help="Number of classes (must match YAML nc)")

    ap.add_argument("--splits", default="train,val,test", help="Comma splits to sanity-check (default: train,val,test)")
    ap.add_argument("--print-paths", action="store_true", help="Print missing/empty/invalid paths (default off)")

    # training toggle
    ap.add_argument("--train", action="store_true", help="Actually train (if omitted, just sanity+eval)")
    ap.add_argument("--base-model", default="yolov8n.pt", help="Base model or pretrained weights for training")

    # evaluation weights (used if --train is NOT set, or to override)
    ap.add_argument("--weights", default=None, help="Weights to evaluate (e.g., runs/detect/.../best.pt). If --train, ignored unless --eval-weights is set.")
    ap.add_argument("--eval-weights", default=None, help="If set, evaluate this weights even after training (override).")

    ap.add_argument("--eval-split", default="test", choices=["train", "val", "test"], help="Which split to evaluate on (default: test)")

    # common YOLO args
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="Ultralytics device string: '' (auto), 'cpu', '0', 'mps', etc.")

    ap.add_argument("--project", default="runs/detect", help="Training output base folder")
    ap.add_argument("--name", default="fpv_gate_train", help="Training run name")

    args = ap.parse_args()

    data_root = args.data_root
    data_yaml = args.data_yaml
    nc = int(args.nc)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = ["train", "val", "test"]

    # 1) sanity check
    print("Running sanity checks...")
    total_invalid = 0
    total_missing = 0
    for sp in splits:
        _, missing, _, inv = sanity_split(data_root, sp, nc=nc, print_paths=args.print_paths)
        total_invalid += inv
        total_missing += missing

    if total_missing > 0 or total_invalid > 0:
        print("\n[STOP] Fix invalid label lines above before training/eval.")
        sys.exit(2)

    # 2) train optional
    best_weights = None
    if args.train:
        print("\nTraining enabled...")
        best_weights = run_train(
            data_yaml=data_yaml,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
        print(f"\nTraining complete. Best weights: {best_weights}")

    # 3) choose weights for eval
    eval_weights = None
    if args.eval_weights:
        eval_weights = args.eval_weights
    elif args.train and best_weights:
        eval_weights = best_weights
    else:
        eval_weights = args.weights

    if not eval_weights:
        print("\n[STOP] No weights to evaluate. Provide --weights or run with --train.")
        sys.exit(3)

    if not os.path.exists(eval_weights):
        print(f"\n[STOP] Weights not found: {eval_weights}")
        sys.exit(4)

    # 4) eval
    print("\nRunning evaluation...")
    run_eval(
        weights=eval_weights,
        data_yaml=data_yaml,
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
