import cv2
import numpy as np
import math


class BBoxCornerExtractor:

    def __init__(self):
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
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(gray, 60, 180)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=20,                     # lower threshold
            minLineLength=min(crop.shape) * 0.15,  # shorter lines allowed
            maxLineGap=40                     # allow bigger gaps
        )

        print("Lines:", None if lines is None else len(lines))
        if lines is None:
            return None
        
        debug_lines = crop.copy()

        for l in lines:
            xA,yA,xB,yB = l[0]
            cv2.line(debug_lines, (xA,yA), (xB,yB), (0,0,255), 2)

        cv2.imshow("hough_lines", debug_lines)

        vertical = []
        horizontal = []

        for l in lines:
            xA,yA,xB,yB = l[0]
            dx = xB - xA
            dy = yB - yA

            if abs(dx) < abs(dy):  # vertical-ish
                vertical.append((xA,yA,xB,yB))
            else:
                horizontal.append((xA,yA,xB,yB))

        if len(vertical) < 2 or len(horizontal) < 2:
            return None

        # take leftmost and rightmost vertical lines
        vertical = sorted(vertical, key=lambda l: (l[0]+l[2])/2)
        left_line = vertical[0]
        right_line = vertical[-1]

        # take topmost and bottommost horizontal lines
        horizontal = sorted(horizontal, key=lambda l: (l[1]+l[3])/2)
        top_line = horizontal[0]
        bottom_line = horizontal[-1]

        def intersect(l1, l2):
            x1,y1,x2,y2 = l1
            x3,y3,x4,y4 = l2

            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                return None

            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
            return [px+x1*0 + x1*0 + x1*0 + x1*0 + x1*0 + x1*0 + x1*0, py]

        # compute corners
        tl = intersect(left_line, top_line)
        tr = intersect(right_line, top_line)
        br = intersect(right_line, bottom_line)
        bl = intersect(left_line, bottom_line)

        if None in (tl,tr,br,bl):
            return None

        pts = np.array([tl,tr,br,bl], dtype=np.float32)

        # shift back to full frame coords
        pts[:,0] += x1
        pts[:,1] += y1

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