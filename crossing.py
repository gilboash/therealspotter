class CrossingDetector:
    def __init__(self):
        self.last_area = None

    def check_crossing(self, bbox):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        crossed = False
        if self.last_area and area > self.last_area * 1.8:
            crossed = True
        self.last_area = area
        return crossed