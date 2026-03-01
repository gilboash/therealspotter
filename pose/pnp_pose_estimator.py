import math
import numpy as np
import cv2

class BBoxPnPEstimator:

    def __init__(self, gate_width_m, gate_height_m, hfov_deg):
        self.gate_w = gate_width_m
        self.gate_h = gate_height_m
        self.hfov = math.radians(hfov_deg)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        w = self.gate_w / 2.0
        h = self.gate_h / 2.0

        self.object_points = np.array([
            [-w, -h, 0],
            [ w, -h, 0],
            [ w,  h, 0],
            [-w,  h, 0],
        ], dtype=np.float32)


    def _build_camera_matrix(self, frame_w, frame_h):
        if self.camera_matrix is not None:
            return

        fx = (frame_w / 2.0) / math.tan(self.hfov / 2.0)
        fy = fx
        cx = frame_w / 2.0
        cy = frame_h / 2.0

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)


    def estimate_from_corners(self, corners, frame_w, frame_h):

        self._build_camera_matrix(frame_w, frame_h)

        if corners is None or len(corners) != 4:
            return None

        image_points = np.asarray(corners, dtype=np.float32)

        image_points = image_points.reshape(-1, 2)

        if image_points.shape != (4, 2):
            return None

        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_to_euler(R)

        return {
            "yaw_deg": math.degrees(yaw),
            "pitch_deg": math.degrees(pitch),
            "roll_deg": math.degrees(roll),
            "z_m": float(tvec[2]),
        }


    def _rotation_to_euler(self, R):
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2(R[1, 0], R[0, 0])
            roll  = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2(-R[0, 1], R[1, 1])
            roll  = 0

        return yaw, pitch, roll