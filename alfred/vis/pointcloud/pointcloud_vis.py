#
# Copyright (c) 2020 JinTian.
#
# This file is part of alfred
# (see http://jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""
showing 3d point cloud using open3d
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from alfred.fusion.common import compute_3d_box_lidar_coords
try:
    from open3d import *
    import open3d as o3d
except ImportError:
    print('importing 3d_vis in alfred-py need open3d installed.')
    exit(0)


def draw_pointclouds_boxes_o3d(pointcloud, boxes_3d, line_color=[0, 1, 0]):
    """
    draw boxes3d on pointcloud, a typical boxes_3d format:
    [[4.481686, 5.147319, -1.0229858, 1.5728549, 3.646751, 1.5121397, 1.5486346],
       [-2.5172017, 5.0262384, -1.0679419, 1.6241353,
           4.0445814, 1.4938312, 1.620804]]
    which is:
    [[x,y,z,w,h,l,roy]]
    """
    geometries = []

    pcs = np.array(pointcloud[:, :3])
    pcobj = o3d.geometry.PointCloud()
    pcobj.points = o3d.utility.Vector3dVector(pcs)
    geometries.append(pcobj)

    # append boxes to geometries
    for p in boxes_3d:
        xyz = np.array([p[: 3]])
        hwl = np.array([p[3: 6]])
        r_y = [p[6]]
        pts3d = compute_3d_box_lidar_coords(
            xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)

        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [line_color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts3d[0])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

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

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_size = 2
    opt.line_width = 3
    # opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()


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
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_size = 2
    opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()
