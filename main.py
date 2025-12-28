import cv2
import argparse
from detector import GateDetector

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "runs/detect/train4/weights/best.pt"

DETECTOR_PARAMS = dict(
    conf=0.35,
    min_area_ratio=0.002,
    aspect_min=0.3,
    aspect_max=3.5,
    persist_frames=3,
    debug=True
)

# ============================================================
# VISUALIZATION
# ============================================================

def draw_boxes(frame, detections, color, label, thickness=2):
    h, w = frame.shape[:2]
    frame_area = h * w

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        area = (x2 - x1) * (y2 - y1)
        area_pct = area / frame_area
        conf = d.get("conf", None)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text = label
        if conf is not None:
            text += f" {conf:.2f}"
        text += f" {area_pct*100:.1f}%"

        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

def draw_legend(frame):
    y = 20
    items = [
        ("RAW (YOLO)", (0, 0, 255)),
        ("FILTERED", (0, 255, 255)),
        ("STABLE (RACE)", (0, 255, 0)),
    ]
    for text, color in items:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 20

# ============================================================
# IOU helper for cumulative counting
# ============================================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["calib", "learn", "race"],
        default="learn",
        help="Operating mode"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional MP4 file"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        print("❌ Failed to open video source")
        return

    detector = GateDetector(
        model_path=MODEL_PATH,
        **DETECTOR_PARAMS
    )

    # cumulative stable gate memory
    cumulative_gates = []

    print(f"▶ Mode: {args.mode.upper()}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = detector.detect(frame)

        raw = out["raw"]
        filtered = out["filtered"]
        stable = out["stable"]

        # ====================================================
        # UPDATE CUMULATIVE STABLE GATES
        # ====================================================
        for det in stable:
            if not any(iou(det["bbox"], g) > 0.5 for g in cumulative_gates):
                cumulative_gates.append(det["bbox"])

        # ====================================================
        # DRAWING BY MODE
        # ====================================================

        if args.mode == "calib":
            draw_boxes(frame, raw, (0, 0, 255), "RAW", thickness=2)
            cv2.putText(frame,
                        f"Gates RAW={len(raw)}",
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        elif args.mode == "learn":
            draw_boxes(frame, raw, (0, 0, 255), "RAW", thickness=1)
            draw_boxes(frame, filtered, (0, 255, 255), "FILTERED", thickness=2)
            draw_boxes(frame, stable, (0, 255, 0), "STABLE", thickness=3)
            draw_legend(frame)

            # show counts
            cv2.putText(frame,
                        f"Gates: RAW={len(raw)} FILTERED={len(filtered)} STABLE={len(stable)} CUM={len(cumulative_gates)}",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        elif args.mode == "race":
            draw_boxes(frame, stable, (0, 255, 0), "GATE", thickness=3)
            cv2.putText(frame,
                        f"Stable Gates: {len(stable)}  Cumulative: {len(cumulative_gates)}",
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        # overlay mode
        cv2.putText(
            frame,
            f"MODE: {args.mode.upper()}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("FPV Spotter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
