import numpy as np
from typing import Any

def depth_to_xyz(depth: np.ndarray,
                 K: np.ndarray,
                 min_depth: int = 100,
                 max_depth: int = 5_000,
                 max_points: int = 5_000) -> np.ndarray:
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mask = (depth > min_depth) & (depth < max_depth)

    if mask.sum() < 100:
        return np.empty((0, 3))

    z = depth[mask].astype(float) / 1_000.0
    x = (xs[mask] - K[0, 2]) * z / K[0, 0]
    y = (ys[mask] - K[1, 2]) * z / K[1, 1]
    points = np.vstack((x, y, z)).T

    # Random down‑sample to tame memory
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    return points[np.isfinite(points).all(1)]
