# rgbd_slam/backend/loop_closure.py
import cv2

class LoopClosureDetector:
    def __init__(self):
        self.keyframes = []
        self.indices = []
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    def add_keyframe(self, gray_img, idx):
        self.keyframes.append(self.sift.detectAndCompute(gray_img, None))
        self.indices.append(idx)

    def detect_loop(self, curr_gray, curr_idx, threshold=0.75, min_matches=40):
        if len(self.keyframes) < 3:
            return None

        kp2, des2 = self.sift.detectAndCompute(curr_gray, None)

        for (kp1, des1), idx in zip(self.keyframes[:-5], self.indices[:-5]):
            if des1 is None or des2 is None:
                continue
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < threshold * n.distance]
            if len(good) > min_matches:
                return idx
        return None
