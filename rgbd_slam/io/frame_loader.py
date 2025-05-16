from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple

class FrameLoader:
    """Load RGB + depth PNGs, keep them in memory (or yield lazily)."""

    def __init__(self, rgb_dir: Path, depth_dir: Path) -> None:
        self.rgb_dir   = rgb_dir
        self.depth_dir = depth_dir
        self._verify_folders()

    # ---------- public API ----------
    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        rgb_files   = sorted(self.rgb_dir.glob("*"))
        depth_files = sorted(self.depth_dir.glob("*"))
        rgb_list:   List[np.ndarray] = []
        depth_list: List[np.ndarray] = []

        for rgb_path, depth_path in zip(rgb_files, depth_files):
            rgb  = cv2.imread(str(rgb_path))
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

            if rgb is None or depth is None or rgb.size == 0 or depth.size == 0:
                print(f"[WARN] Skipping corrupt pair {rgb_path.name} / {depth_path.name}")
                continue

            rgb_list.append(rgb)
            depth_list.append(depth)

        print(f"[INFO] Loaded {len(rgb_list)} RGB‑D frames")
        return rgb_list, depth_list

    # ---------- helpers ----------
    def _verify_folders(self) -> None:
        for p, tag in [(self.rgb_dir, "RGB"), (self.depth_dir, "depth")]:
            if not p.exists():
                raise FileNotFoundError(f"{tag} folder ‘{p}’ not found")
