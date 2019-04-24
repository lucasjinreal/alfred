import os

import sys
import numpy as np
from alfred.vis.image.common import get_unique_color_by_id
from alfred.fusion.kitti_fusion import LidarCamCalibData, \
    load_pc_from_file, cam3d_to_pixel, lidar_pt_to_cam0_frame
from alfred.fusion.common import draw_3d_box, compute_3d_box, center_to_corner_3d
import cv2


img_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/000011.png')
v_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/000011.bin')
calib_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), './data/000011.txt')
frame_calib = LidarCamCalibData(calib_f=calib_f)

res = [[5.06, 1.43, 12.42, 1.90, 0.42, 1.04,  0.68],
       [-5.12, 1.85, 4.13, 1.50, 1.46, 3.70,  1.56],
       [-4.95, 1.83, 26.64, 1.86, 1.57, 3.83,  1.55]]

img = cv2.imread(img_f)

for p in res:
    xyz = np.array([p[: 3]])

    c2d = cam3d_to_pixel(xyz, frame_calib)
    if c2d is not None:
        cv2.circle(img, (int(c2d[0]), int(c2d[1])), 8, (0, 0, 255), -1)

    # hwl -> lwh
    lwh = np.array([p[3: 6]])[:, [2, 1, 0]]
    r_y = p[6]
    pts3d = compute_3d_box(xyz[0], lwh[0], r_y)

    pts2d = []
    for pt in pts3d:
        coords = cam3d_to_pixel(pt, frame_calib)
        if coords is not None:
            pts2d.append(coords[:2])
    pts2d = np.array(pts2d)
    draw_3d_box(pts2d, img)

cv2.imshow('rr', img)
cv2.imwrite('result.png', img)
cv2.waitKey(0)
