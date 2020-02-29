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
import cv2
import numpy as np


def draw_3d_box(pts, img, color=(255, 0, 255), thickness=1):
    """
    Given 8 points of a 3D Bounding Box, draw it on image
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    the points assume above order

    :param pts:
    :param img:
    :param color:
    :param thickness:
    :return:
    """
    # currently, we skip box which not has 8 points
    pts = np.array(pts, dtype=np.int)
    if pts.shape != (8, 2):
        return
        # clockwise face idx
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        # 3, 2, 1, 0
        f = face_idx[ind_f]

        # indicates the direction
        for j in range(4):
            if ind_f == 0 and j == 0:
                cv2.line(img, (pts[f[0], 0], pts[f[0], 1]),
                         (pts[f[1], 0], pts[f[1], 1]), (255, 255, 0),
                         thickness, lineType=cv2.LINE_AA)
            cv2.line(img, (pts[f[j], 0], pts[f[j], 1]),
                     (pts[f[(j + 1) % 4], 0], pts[f[(j + 1) % 4], 1]), color,
                     thickness, lineType=cv2.LINE_AA)


def compute_3d_box_cam_coords(xyz, lwh, rotation_y):
    """
    KITTI camera coordinates using -y as up
    this only works on camera coordinates xyz
    center
    dim
    rotation

    Algorithm: supports a 3d box at center (0, 0, 0), using r_y we can get a Rotate matrix
    calculate the new 3d box after rotate by Rotate operation.
    :param xyz:
    :param lwh:
    :param rotation_y:
    :return:
    """
    location = xyz
    dim = lwh
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[0], dim[1], dim[2]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    # y_corners = [0, 0, 0, 0, h, h, h, h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def compute_3d_box_lidar_coords(centers,
                                dims,
                                angles=None,
                                origin=(0.5, 0.5, 0.5),
                                axis=2):
    corners = _corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = _rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


# ------------------ Should not calling directly
def _rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def _corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners
