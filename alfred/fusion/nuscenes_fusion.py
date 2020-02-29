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

Fusion for nuScenes dataset

"""
import numpy as np
from pyquaternion import Quaternion


def project_cam_coords_to_pixel(pts3d_cam, intrinsic):
    """
    project pts3d_cam on image
    such as:

    [[-1.9, 0.12, 37]] -> [[222, 345]]
    """
    intrinsic = np.array(intrinsic)
    pts3d_cam = np.array(pts3d_cam)
    if pts3d_cam.shape[0] != 3:
        # if not hstack, try to transpose it
        pts3d_cam = pts3d_cam.T
    assert pts3d_cam.shape[0] == 3, 'pts3d_cam must be 3 rows.'
    assert intrinsic.shape == (3, 3), 'intrinsic should be 3x3.'
    a = np.dot(intrinsic, pts3d_cam)
    a = a/a[-1, :]
    return a.T[:, :2]

# def compute_3d_box_cam_coords_nuscenes(xyz, lwh, quarternion):
#     """
#         nuScenes camera coordinates using -y as up
#         using quarternion represents rotation of box
#     """
#     # we get rotation_y from quarternion first
#     quarternion = Quaternion(axis=quarternion[1:], angle=quarternion[0])
#     trans_m = transform_matrix(xyz, quarternion)
#     l, w, h = lwh[0], lwh[1], lwh[2]
#     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
#     y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
#     # y_corners = [0, 0, 0, 0, h, h, h, h]
#     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

#     corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
#     corners4d = np.ones((corners.shape[0]+1, corners.shape[1]))
#     corners4d[:-1, :] = corners
#     print('transformation matrix: ', trans_m)
#     print('corners4d: ', corners4d)
#     corners4d_trans = np.dot(trans_m, corners4d)
#     print('corners4d_trans: ', corners4d_trans)
#     return corners4d_trans


def compute_3d_box_cam_coords_nuscenes(xyz, lwh, quarternion):
    """
        nuScenes camera coordinates using -y as up
        using quarternion represents rotation of box

        only calculate rotation_y?: arcsin(2(wy-zx))
    """
    # we get rotation_y from quarternion first
    if isinstance(quarternion, Quaternion):
        l, w, h = lwh[0], lwh[1], lwh[2]
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        # y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        y_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

        rotation_y = np.pi/2 - quarternion.radians
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        corners_trans = np.dot(R, corners)
        corners_trans += [[i] for i in xyz]
        return corners_trans
    else:
        raise ValueError('quarternion must be a Quaternion object, make sure '\
            'you using pyquarternion.')


def load_pc_from_file(pc_f):
    # nuScenes lidar is 5 digits one line (last one the ring index)
    return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 5])