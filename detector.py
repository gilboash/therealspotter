from ultralytics import YOLO
import numpy as np
from collections import deque

class GateDetector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf=0.25,                  # MUCH lower
        min_area_ratio=0.002,       # allow far gates
        aspect_min=0.2,             # very permissive
        aspect_max=5.0,
        persist_frames=3,
        min_persist_hits=2,         # tolerate misses
        debug=True
    ):
        """
        Relaxed FPV gate detector with debug support
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.min_area_ratio = min_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.persist_frames = persist_frames
        self.min_persist_hits = min_persist_hits
        self.debug = debug

        self.history = deque(maxlen=persist_frames)

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        frame_area = frame.shape[0] * frame.shape[1]

        raw = []
        filtered = []

        # ----------------------------
        # 1. Collect RAW detections
        # ----------------------------
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = w * h
            aspect = w / h
            score = float(box.conf)

            det = {
                "bbox": (x1, y1, x2, y2),
                "conf": score,
                "area_ratio": area / frame_area,
                "aspect": aspect,
                "cls": int(box.cls)
            }

            raw.append(det)

            # ----------------------------
            # 2. SOFT filtering
            # ----------------------------
            if area < self.min_area_ratio * frame_area:
                continue
            if not (self.aspect_min <= aspect <= self.aspect_max):
                continue

            filtered.append(det)

        # ----------------------------
        # 3. Temporal persistence (soft)
        # ----------------------------
        self.history.append(filtered)
        stable = []

        if len(self.history) >= 2:
            for det in filtered:
                hits = sum(
                    any(self.iou(det["bbox"], old["bbox"]) > 0.4 for old in hist)
                    for hist in self.history
                )
                if hits >= self.min_persist_hits:
                    stable.append(det)
        else:
            stable = filtered

        # ----------------------------
        # 4. Return debug-rich output
        # ----------------------------
        if self.debug:
            return {
                "raw": raw,
                "filtered": filtered,
                "stable": stable
            }

        return stable

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0:
            return 0.0

        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter)
