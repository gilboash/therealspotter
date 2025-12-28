from ultralytics import YOLO
import numpy as np
from collections import deque

class GateDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.65, min_area_ratio=0.01, aspect_min=0.5, aspect_max=2.0, persist_frames=3):
        """
        model_path: path to YOLO model (can be custom fpv_gate.pt)
        conf: confidence threshold
        min_area_ratio: reject boxes smaller than this fraction of frame area
        aspect_min/aspect_max: reject extreme aspect ratios
        persist_frames: require detection to persist N frames to reduce false positives
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.min_area_ratio = min_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.persist_frames = persist_frames

        # history of detected boxes: deque of lists
        self.history = deque(maxlen=persist_frames)

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        frame_area = frame.shape[0] * frame.shape[1]

        filtered = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect = w / max(h, 1)

            # filter tiny or weird boxes
            if area < self.min_area_ratio * frame_area:
                continue
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue

            # Accept this detection
            filtered.append({
                "bbox": (x1, y1, x2, y2),
                "embedding": box.cls.cpu().numpy()
            })

        # Temporal stability: keep only boxes that appear persist_frames times
        self.history.append(filtered)
        stable = []
        if len(self.history) == self.persist_frames:
            # count consistent detections
            for det in filtered:
                count = sum(
                    any(self.iou(det['bbox'], old_det['bbox']) > 0.5 for old_det in h)
                    for h in list(self.history)
                )
                if count == self.persist_frames:
                    stable.append(det)
        else:
            stable = filtered  # warm-up, just output

        return stable

    @staticmethod
    def iou(boxA, boxB):
        # Compute Intersection over Union for 2 boxes
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
