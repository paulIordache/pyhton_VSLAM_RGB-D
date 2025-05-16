#!/usr/bin/env python3
"""
Entry-point.  Usage:

    python run_slam.py --rgb data/rgb --depth data/depth
"""
import argparse
import numpy as np
from pathlib import Path
import cv2

from rgbd_slam.config import (CameraIntrinsics, RGB_DIR, DEPTH_DIR, DEBUG_DIR,
                              SAMPLE_INTERVAL, DAMPING, MAX_POINT_CLOUD_SIZE)
from rgbd_slam.io.frame_loader        import FrameLoader
from rgbd_slam.vision.feature_matching import FeatureMatcher
from rgbd_slam.vision.pose_estimation  import PoseEstimator
from rgbd_slam.pointcloud.depth_to_pointcloud import depth_to_xyz
from rgbd_slam.viz.trajectory_plotter  import TrajectoryPlotter
from rgbd_slam.viz.pointcloud_viz      import show as show_pointcloud
from rgbd_slam.backend.loop_closure    import LoopClosureDetector

def main() -> None:
    args = _cli()
    intr = CameraIntrinsics()
    loader = FrameLoader(args.rgb_dir, args.depth_dir)
    matcher = FeatureMatcher()
    estimator = PoseEstimator(intr.K)
    plotter = TrajectoryPlotter()
    loop_closure = LoopClosureDetector()

    rgb_frames, depth_frames = loader.load()
    if not rgb_frames:
        raise SystemExit("No frames found!")

    trajectory   = []
    cumulative_T = np.eye(4)
    all_points   = []

    for i in range(1, len(rgb_frames), SAMPLE_INTERVAL):
        print(f"[INFO] Processing {i}/{len(rgb_frames)-1}")
        prev_rgb, curr_rgb = rgb_frames[i-1], rgb_frames[i]
        prev_pts, curr_pts = matcher.match(prev_rgb, curr_rgb)

        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_BGR2GRAY)
        loop_idx = loop_closure.detect_loop(curr_gray, i)
        if loop_idx is not None:
            print(f"[LOOP] Detected loop between frame {i} and {loop_idx}!")

        if len(prev_pts) < 8:
            trajectory.append(cumulative_T[:3, 3])
            plotter.update(trajectory)
            loop_closure.add_keyframe(curr_gray, i)
            continue

        R, t, _ = estimator.estimate(prev_pts, curr_pts)
        if np.linalg.norm(t) > 1.0:
            t *= 1.0 / np.linalg.norm(t)

        delta = np.eye(4)
        delta[:3, :3] = R * DAMPING + np.eye(3) * (1 - DAMPING)
        delta[:3,  3] = (t.flatten() * DAMPING)
        cumulative_T = cumulative_T @ delta

        trajectory.append(cumulative_T[:3, 3])
        plotter.update(trajectory)
        loop_closure.add_keyframe(curr_gray, i)

        if i % 20 == 1:
            pts = depth_to_xyz(depth_frames[i-1], intr.K,
                               max_points=MAX_POINT_CLOUD_SIZE)
            if len(pts):
                homog = np.hstack((pts, np.ones((len(pts), 1))))
                all_points.append((cumulative_T @ homog.T).T[:, :3])

    _save_np("trajectory.txt", np.vstack(trajectory))
    if all_points:
        pc = np.vstack(all_points)
        pc = _filter_outliers(pc)
        _save_np("point_cloud.txt", pc)
        show_pointcloud(pc)

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dir",   type=Path, default=RGB_DIR,   help="RGB folder")
    p.add_argument("--depth-dir", type=Path, default=DEPTH_DIR, help="depth folder")
    return p.parse_args()

def _save_np(fname: str, arr: np.ndarray) -> None:
    np.savetxt(fname, arr)
    print(f"[INFO] wrote {fname}")

def _filter_outliers(pc: np.ndarray) -> np.ndarray:
    mu, sigma = pc.mean(0), pc.std(0)
    dist = np.linalg.norm(pc - mu, axis=1)
    mask = dist < 3 * sigma.mean()
    print(f"[INFO] point cloud filtered: {pc.shape[0]} → {mask.sum()}")
    return pc[mask]

if __name__ == "__main__":
    main()