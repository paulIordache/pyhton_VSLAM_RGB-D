import numpy as np
import open3d as o3d

def show(points: np.ndarray, window: str = "Point cloud") -> None:
    if len(points) == 0:
        return
    pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points.astype(np.float32)))
    pcd.paint_uniform_color([0.5, 0.5, 0.8])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window, width=1280, height=720)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.run()
    vis.destroy_window()
