import cv2
import numpy as np
import os
from typing import Tuple

class FeatureMatcher:
    def __init__(self, nfeatures: int = 3000) -> None:
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        index_params = dict(algorithm=1, trees=5)  # KD-tree
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, prev: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create debug directory if it doesn't exist
        os.makedirs("debug", exist_ok=True)

        # Preprocess images
        prev_g = self._prep(prev)
        curr_g = self._prep(curr)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(prev_g, None)
        kp2, des2 = self.sift.detectAndCompute(curr_g, None)

        # Draw and save keypoints regardless of match success
        prev_kp_img = cv2.drawKeypoints(prev_g, kp1, None)
        curr_kp_img = cv2.drawKeypoints(curr_g, kp2, None)
        cv2.imwrite("debug/debug_prev_keypoints.png", prev_kp_img)
        cv2.imwrite("debug/debug_curr_keypoints.png", curr_kp_img)

        if des1 is None or des2 is None:
            print("[DEBUG] One of the descriptor sets is None.")
            return np.array([]), np.array([])

        # Match descriptors using FLANN + Lowe's ratio test
        raw_matches = self.flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]

        print(f"[DEBUG] Total raw matches: {len(raw_matches)}, Good matches: {len(good)}")

        if len(good) < 8:
            print("[DEBUG] Not enough good matches (< 8).")
            return np.array([]), np.array([])

        # Draw matches and save
        match_img = cv2.drawMatches(prev_g, kp1, curr_g, kp2, good, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("debug/debug_matches.png", match_img)

        # Return matched keypoints
        pts_prev = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts_prev, pts_curr

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        grey = cv2.equalizeHist(grey)
        return cv2.GaussianBlur(grey, (5, 5), 0)
