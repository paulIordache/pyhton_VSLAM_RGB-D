import cv2
import numpy as np
from typing import Tuple

class PoseEstimator:
    def __init__(self, K: np.ndarray) -> None:
        self.K = K

    def estimate(self,
                 pts_prev: np.ndarray,
                 pts_curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pts_prev) < 8:
            return np.eye(3), np.zeros((3, 1)), np.array([])

        E, _ = cv2.findEssentialMat(pts_curr, pts_prev, self.K,
                                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return np.eye(3), np.zeros((3, 1)), np.array([])

        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, self.K)

        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))

        pts_prev_h = pts_prev.T
        pts_curr_h = pts_curr.T

        pts4d = cv2.triangulatePoints(P1, P2, pts_prev_h, pts_curr_h)
        pts3d = (pts4d[:3] / pts4d[3]).T
        return R, t, pts3d
