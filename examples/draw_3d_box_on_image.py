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
import os

import sys
import numpy as np
from alfred.vis.image.common import get_unique_color_by_id
from alfred.fusion.kitti_fusion import LidarCamCalibData, \
    load_pc_from_file, lidar_pts_to_cam0_frame, lidar_pt_to_cam0_frame
from alfred.fusion.common import draw_3d_box, compute_3d_box_lidar_coords
import cv2

# from 2011_09_26/2011_09_26_drive_0051_sync

img_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/000011.png')
v_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/000011.bin')
calib_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/000011.txt')

frame_calib = LidarCamCalibData(calib_f=calib_f)

res = [[4.481686, 5.147319, -1.0229858, 1.5728549, 3.646751, 1.5121397, 1.5486346],
       [-2.5172017, 5.0262384, -1.0679419, 1.6241353, 4.0445814, 1.4938312, 1.620804],
       [1.1783253, -2.9209857, -0.9852259, 1.5852798, 3.7360613, 1.4671413, 1.5811548],
       [12.925569, -4.9808474, -0.71562666, 0.5328532, 0.89768076, 1.7436955, 0.7869441],
       [-9.657954, -2.9310253, -0.9663244, 1.6315838, 4.0691543, 1.4506648, 4.7061768],
       [-7.734651, 4.928315, -1.3513744, 1.7096852, 4.41021, 1.4849466, 1.5580404],
       [-21.06287, -6.378005, -0.6494193, 0.58654386, 0.67096156, 1.7274126, 1.5062331],
       [-12.977588, 4.7324443, -1.2884868, 1.6366509, 3.993301, 1.4792416, 1.5961027],
       [27.237848, 4.973592, -0.63590205, 1.6796488, 4.1773257, 1.8397285, 1.5534456],
       [-15.21727, -3.3323386, -1.1841949, 1.5691711, 3.7851675, 1.4302691, 1.4623685],
       [-8.560741, -15.309304, -0.40493315, 1.5614295, 3.6039133, 1.4802926, 3.685232],
       [-28.535696, 1.8784677, -1.349385, 1.8589652, 4.6122866, 2.0191495, 4.708105],
       [22.139666, -19.737762, -0.74519694, 0.52543664, 1.7905389, 1.684143, -0.26117292],
       [-4.4033785, -2.856424, -0.95746094, 1.7221596, 4.5044794, 1.6574095, 1.5402203],
       [7.085311, -12.124656, -0.7908472, 1.605196, 4.036379, 1.4904786, 3.1525888],
       [-17.75546, 4.869718, -1.4353731, 1.625128, 4.0645328, 1.4669982, 1.5843123],
       [22.015368, -16.157223, -0.97120696, 0.70649695, 1.8466028, 1.6473441, 3.46424],
       [34.445316, -2.0812414, -0.5032885, 0.6895117, 0.8842125, 1.7723539, -1.4539356],
       [-32.120346, 7.0260167, -1.6048443, 0.59323585, 0.7810404, 1.7134606, 0.9840808],
       [11.191077, -20.68808, -0.3166721, 2.1275487, 6.112693, 2.4575462, 4.6473494],
       [-0.18853411, -11.496099, -0.723109, 1.6154484, 3.9286208, 1.5749075, 3.0955489],
       [7.4211736, -7.1129866, -1.355744, 1.5750822, 3.9536934, 1.4568869, -0.6677291],
       [16.404984, 7.875185, -0.9816911, 0.64251673, 0.63132536, 1.7938845, 1.0830851],
       [20.704462, -21.648046, -0.99220616, 1.5985962, 3.830404, 1.521529, 3.0288131],
       [-34.060417, -1.6139596, -1.1061747, 0.73393285, 0.8841753, 1.7669718, 4.5250244],
       [-9.143257, -8.996165, -0.9218217, 1.5279316, 3.592435, 1.4721779, 0.85066897],
       [-31.856539, -2.953291, -1.4160485, 0.67631316, 0.86612713, 1.7683575, 3.113426],
       [-29.955063, -4.6513176, -1.2724423, 1.5479406, 3.5412807, 1.463421, 0.11858773],
       [10.639572, 11.339079, -0.35397023, 0.6703583, 0.57711476, 1.7787935, 4.486712],
       [-11.947865, -21.075172, -0.32996762, 1.5983682, 3.945621, 1.4992962, 1.6880405],
       [-17.38843, -6.5131726, -0.07191068, 0.6577756, 0.7161297, 1.8168749, 1.8645211],
       [2.0013125, -16.632671, -0.54558295, 0.54916567, 1.8482145, 1.7980447, 5.3003416]]

pcs = load_pc_from_file(v_f)
img = cv2.imread(img_f)


for p in res:
    xyz = np.array([p[: 3]])

    c2d = lidar_pt_to_cam0_frame(xyz, frame_calib)
    if c2d is not None:
        cv2.circle(img, (int(c2d[0]), int(c2d[1])), 3, (0, 255, 255), -1)

    hwl = np.array([p[3: 6]])
    r_y = [p[6]]
    print('xyz: {}, whl: {}, r_y: {}'.format(xyz, hwl, r_y))
    pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)

    pts2d = []
    for pt in pts3d[0]:
        coords = lidar_pt_to_cam0_frame(pt, frame_calib)
        if coords is not None:
            pts2d.append(coords[:2])
    pts2d = np.array(pts2d)
    draw_3d_box(pts2d, img)

cv2.imshow('rr', img)
cv2.imwrite('result.png', img)
cv2.waitKey(0)
