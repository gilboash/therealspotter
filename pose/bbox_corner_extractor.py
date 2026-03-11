import cv2
import numpy as np
import math

from mobile_sam import sam_model_registry, SamPredictor
import torch



class BBoxCornerExtractor:

    def __init__(self):
        sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
        if torch.backends.mps.is_available():
            sam.to(device="mps")   # or "cuda"
            print("run model mps - mac")
        else:
            sam.to(device="cpu")   # or "cuda"
            print("run model cpu")

        predictor = SamPredictor(sam)

        self.sam_predictor = predictor
        pass


    def _filter_and_assign_corners(self, corners, bbox):

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return None

        bbox_corners = np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ], dtype=np.float32)

        assigned = []

        for bc in bbox_corners:
            best = None
            best_dist = float("inf")

            for pt in corners:
                d = np.linalg.norm(np.array(pt) - bc)
                if d < best_dist:
                    best_dist = d
                    best = pt

            # Allow up to 35% of bbox diagonal
            max_allowed = 0.35 * math.hypot(w, h)

            if best is None or best_dist > max_allowed:
                return None

            assigned.append(best)

        return np.array(assigned, dtype=np.float32)

    def extract_corners(self, frame, bbox):
        import numpy as np
        import cv2

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            print("Empty crop")
            return None

        predictor = self.sam_predictor

        # ---- SAM segmentation ----

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        h, w = crop.shape[:2]

        input_box = np.array([5, 5, w-5, h-5])
        input_point = np.array([[w/2, h/2]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True
        )

        best_mask = None
        best_score = -1

        for m, s in zip(masks, scores):

            area = np.sum(m)

            # reject tiny masks and full-image masks
            if 0.05 * w * h < area < 0.9 * w * h:

                if s > best_score:
                    best_mask = m
                    best_score = s

        if best_mask is None:
            print("SAM failed to produce valid mask")
            return None

        mask = (best_mask.astype(np.uint8) * 255)

        # ---- DEBUG: show mask ----

        cv2.imshow("SAM mask", mask)

        segmented = cv2.bitwise_and(crop, crop, mask=mask)
        cv2.imshow("SAM segmented", segmented)

        # ---- find largest contour ----

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("No contour found")
            return None

        cnt = max(contours, key=cv2.contourArea)

        contour_area = cv2.contourArea(cnt)

        if contour_area < 0.02 * w * h:
            print("Rejected: contour too small")
            return None

        # ---- fit rectangle to contour ----

        rect = cv2.minAreaRect(cnt)

        rect_w, rect_h = rect[1]

        if rect_w == 0 or rect_h == 0:
            print("Rejected: invalid rectangle")
            return None

        rect_area = rect_w * rect_h

        if rect_area < 0.02 * w * h:
            print("Rejected: rectangle too small")
            return None

        # ---- extract rectangle corners ----

        box = cv2.boxPoints(rect)

        pts = np.array(box, dtype=np.float32)

        # ---- order corners consistently ----

        center = np.mean(pts, axis=0)

        def angle_from_center(p):
            return np.arctan2(p[1] - center[1], p[0] - center[0])

        pts = np.array(sorted(pts, key=angle_from_center), dtype=np.float32)

        # ---- DEBUG VISUALIZATION ----

        dbg = crop.copy()

        # draw contour
        cv2.drawContours(dbg, [cnt], -1, (0,255,255), 2)

        # draw fitted rectangle
        cv2.polylines(
            dbg,
            [pts.astype(int)],
            True,
            (255,0,0),
            2
        )

        # draw corners
        for p in pts:
            cv2.circle(
                dbg,
                tuple(p.astype(int)),
                6,
                (0,255,0),
                -1
            )

        cv2.imshow("gate_rect", dbg)

        # ---- convert back to frame coordinates ----

        pts[:,0] += x1
        pts[:,1] += y1

        print("Corners accepted")
        print("Corners:", pts)

        return pts


    def _line_length(self, l):
        x1, y1, x2, y2 = l
        return math.hypot(x2 - x1, y2 - y1)

    def _line_intersection(self, l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

        return [px, py]

    def _order_corners(self, pts):
        pts = np.array(pts)

        center = np.mean(pts, axis=0)

        ordered = []
        for p in pts:
            angle = math.atan2(p[1] - center[1], p[0] - center[0])
            ordered.append((angle, p))

        ordered = sorted(ordered, key=lambda x: x[0])
        ordered_pts = [p for _, p in ordered]

        return np.array(ordered_pts, dtype=np.float32)