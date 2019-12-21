"""
showing 3d point cloud using open3d
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
try:
    from open3d import *
except ImportError:
    print('importing 3d_vis in alfred-py need open3d installed.')
    exit(0)


def draw_pcs_open3d(geometries):
    """
    drawing the points using open3d
    it can draw points and linesets
    ```
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(pcs)


    points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    lines = [[0,1],[0,2],[1,3],[2,3],
                [4,5],[4,6],[5,7],[6,7],
                [0,4],[1,5],[2,6],[3,7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)
    draw_pcs_open3d([point_cloud, line_set])
    ```
    """
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
    vis = Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    # opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()
