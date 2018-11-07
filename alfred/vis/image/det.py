"""
draw detection result base on various format

after detection

also include draw 3d box on image
"""
import numpy as np
import cv2
import os

from .common import create_unique_color_uchar


# --------------- Drawing 3d box on image parts --------------
def draw_one_3d_box_cv2(img, box_3d, obj_id_name_map, score, tlwhy_format=False, calib_cam_to_img_p2=None,
                        force_color=None):
    """
    provide a obj id name map like: {1, 'car'}
    id to distinguish with previous object type

    tlwhy means input box are in format: [x, y, z, l, w, h, ry]
    that means we should convert it first.
    :param img:
    :param box_3d:
    :param obj_id_name_map:
    :param score:
    :param tlwhy_format:
    :param calib_cam_to_img_p2:
    :param force_color:
    :return:
    """
    assert isinstance(obj_id_name_map, dict), 'obj_id_name_map must be dict'
    # color = None
    if force_color:
        color = force_color
    else:
        color = create_unique_color_uchar(list(obj_id_name_map.keys())[0])
    if tlwhy_format:
        # transform [x, y, z, l, w, h, ry] to normal box
        assert calib_cam_to_img_p2, 'You should provide calibration matrix, convert camera to image coordinate.'
        center = box_3d[0: 3]
        dims = box_3d[3: 6]
        rot_y = -box_3d[6] / 180 * np.pi
        # alpha / 180 * np.pi + np.arctan(center[0] / center[2])

        converted_box_3d = []
        for i in [1, -1]:
            for j in [1, -1]:
                for k in [0, 1]:
                    point = np.copy(center)
                    point[0] = center[0] + i * dims[1] / 2 * np.cos(-rot_y + np.pi / 2) + \
                               (j * i) * dims[2] / 2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dims[1] / 2 * np.sin(-rot_y + np.pi / 2) + \
                               (j * i) * dims[2] / 2 * np.sin(-rot_y)
                    point[1] = center[1] - k * dims[0]

                    point = np.append(point, 1)
                    point = np.dot(calib_cam_to_img_p2, point)
                    point = point[:2] / point[2]
                    point = point.astype(np.int16)
                    converted_box_3d.append(point)
        print('final box: ', converted_box_3d)
        # box_3d = np.asarray(converted_box_3d)
        box_3d = converted_box_3d
        # print(box_3d.shape)
        for i in range(4):
            point_1_ = box_3d[2 * i]
            point_2_ = box_3d[2 * i + 1]
            cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 1)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i + 2) % 8]
            cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 1)
        return img
    else:
        # assert len(box_3d) == 8, 'every box 3d should have 8 points. if you got 7, you may want tlwhy=True'
        face_idx = np.array([0, 1, 5, 4,  # front face
                             1, 2, 6, 5,  # left face
                             2, 3, 7, 6,  # back face
                             3, 0, 4, 7]).reshape((4, 4))
        # print('start draw...')
        for i in range(4):
            x = np.append(box_3d[0, face_idx[i, ]],
                          box_3d[0, face_idx[i, 0]])
            y = np.append(box_3d[1, face_idx[i, ]],
                          box_3d[1, face_idx[i, 0]])
            # print('x: ', x)
            # print('y: ', y)
            # cv2.line(img, (point_1_, point_1_), (point_2_, point_2_), color, 1)
            pts = np.vstack((x, y)).T
            # filter negative values
            pts = (pts + abs(pts)) / 2
            pts = np.array([pts], dtype=int)
            # print(pts)
            cv2.polylines(img, pts, isClosed=True, color=color, thickness=1)
            if i == 3:
                # add text
                ori_txt = pts[0][1]

        return img
