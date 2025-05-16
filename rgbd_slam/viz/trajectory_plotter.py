import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List

class TrajectoryPlotter:
    def __init__(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.set_title("Camera trajectory")
        self.ax.grid(True)

    def update(self, trajectory: List[np.ndarray]) -> None:
        if not trajectory:
            return
        traj = np.array(trajectory)
        self.ax.clear()
        self.ax.plot(traj[:, 0], traj[:, 2], "b-")
        self.ax.plot(traj[0, 0],  traj[0, 2],  "go", label="start")
        self.ax.plot(traj[-1, 0], traj[-1, 2], "ro", label="current")
        self.ax.legend()
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.set_title("Camera trajectory")
        self.ax.grid(True)
        plt.draw(); plt.pause(0.001)
