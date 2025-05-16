from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0,        self.cx],
                         [0,        self.fy, self.cy],
                         [0,        0,        1     ]], dtype=float)

DATA_DIR      = Path("data")      # ← change if you like
RGB_DIR       = "rgb" / DATA_DIR
DEPTH_DIR     = "depth" / DATA_DIR 
DEBUG_DIR     = Path("debug_output")
MAX_POINT_CLOUD_SIZE = 5_000
SAMPLE_INTERVAL      = 5          # Process every Nth frame
DAMPING              = 0.5        # For pose interpolation
