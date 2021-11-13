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
draw detection result base on various format

after detection

also include draw 3d box on image
"""
import numpy as np
import cv2
import os
import random
from .common import create_unique_color_uchar, get_unique_color_by_id2, colors
from .common import draw_rect_with_style
import warnings
from collections import Counter, OrderedDict
from .common import put_txt_with_newline
from .get_dataset_label_map import coco_label_map_list


def _draw_round_dot_border(img, pt1, pt2, color, thickness, r=2, d=5):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    return img


def draw_one_bbox(image, box, unique_color, thickness):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), unique_color, thickness)
    return image


# ==================== Below are deprecation API =================
def draw_box_without_score(img, boxes, classes=None, is_show=False):
    """
    Draw boxes on image, the box mostly are annotations, not the model predict box
    """
    warnings.warn(
        'this method is deprecated, using visiualize_det_cv2 instead', DeprecationWarning)
    assert isinstance(boxes,
                      np.ndarray), 'boxes must nump array, with shape of (None, 5)\nevery element contains (x1,y1,x2,y2, label)'
    if classes:
        pass
    else:
        height = img.shape[0]
        width = img.shape[1]

        font = cv2.QT_FONT_NORMAL
        font_scale = 0.4
        font_thickness = 1
        line_thickness = 1

        all_cls = []
        for i in range(boxes.shape[0]):
            cls = boxes[i, -1]
            all_cls.append(cls)
            all_cls = set(all_cls)
            unique_color = create_unique_color_uchar(all_cls.index(cls))

            y1 = int(boxes[i, 2])
            x1 = int(boxes[i, 3])
            y2 = int(boxes[i, 4])
            x2 = int(boxes[i, 5])

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          unique_color, line_thickness)

            text_label = '{}'.format(cls)
            (ret_val, base_line) = cv2.getTextSize(
                text_label, font, font_scale, font_thickness)
            text_org = (x1, y1 - 0)

            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line + 2),
                          (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] - 2), unique_color, line_thickness)
            # this rectangle for fill text rect
            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line + 2),
                          (text_org[0] + ret_val[0] + 4,
                           text_org[1] - ret_val[1] - 2),
                          unique_color, -1)
            cv2.putText(img, text_label, text_org, font,
                        font_scale, (255, 255, 255), font_thickness)
        if is_show:
            cv2.imshow('image', img)
            cv2.waitKey(0)
        return img


def visualize_det_cv2(img, detections, classes=None, thresh=0.6, is_show=False, background_id=-1, mode='xyxy'):
    """
    visualize detection on image using cv2, this is the standard way to visualize detections

    new add mode option
    mode can be one of 'xyxy' and 'xywh', 'xyxy' as default

    :param img:
    :param detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
    :param classes:
    :param thresh:
    :param is_show:
    :param background_id: -1
    :param mode:
    :return:
    """
    assert classes, 'from visualize_det_cv2, classes must be provided, each class in a list with' \
                    'certain order.'
    assert isinstance(
        img, np.ndarray), 'from visualize_det_cv2, img must be a numpy array object.'

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.36

    font_thickness = 1
    line_thickness = 1

    for i in range(detections.shape[0]):
        cls_id = int(detections[i, 0])
        if cls_id != background_id:
            score = detections[i, 1]
            if score > thresh:
                unique_color = create_unique_color_uchar(cls_id)
                x1, y1, x2, y2 = 0, 0, 0, 0
                if mode == 'xyxy':
                    x1 = int(detections[i, 2])
                    y1 = int(detections[i, 3])
                    x2 = int(detections[i, 4])
                    y2 = int(detections[i, 5])
                else:
                    x1 = int(detections[i, 2])
                    y1 = int(detections[i, 3])
                    x2 = x1 + int(detections[i, 4])
                    y2 = y1 + int(detections[i, 5])

                cv2.rectangle(img, (x1, y1), (x2, y2),
                              unique_color, line_thickness, cv2.LINE_AA)
                text_label = '{} {:.2f}'.format(classes[cls_id], score)
                (ret_val, _) = cv2.getTextSize(
                    text_label, font, font_scale, font_thickness)
                txt_bottom_left = (x1+4, y1-4)
                cv2.rectangle(img, (txt_bottom_left[0]-4, txt_bottom_left[1] - ret_val[1]-2),
                              (txt_bottom_left[0] + ret_val[0] +
                               2, txt_bottom_left[1]+4),
                              (0, 0, 0), -1)
                cv2.putText(img, text_label, txt_bottom_left, font,
                            font_scale, (237, 237, 237), font_thickness, cv2.LINE_AA)
    if is_show:
        cv2.imshow('image', img)
        cv2.waitKey(0)
    return img


def visualize_det_cv2_style0(img, detections, classes=None, cls_colors=None, thresh=0.3, suit_color=True, blend=False,
                             is_show=False, background_id=-1, mode='xyxy', line_thickness=1,
                             font_scale=0.48, counter_on=False, text_bk=False, counter_pos=(30, 150)):
    """
    visualize detection on image using cv2, this is the standard way to visualize detections

    new add mode option
    mode can be one of 'xyxy' and 'xywh', 'xyxy' as default

    :param img:
    :param detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
    :param classes:
    :param cls_colors:
    :param thresh:
    :param is_show:
    :param background_id: -1
    :param mode:
    :return:
    """
    assert isinstance(
        img, np.ndarray), 'from visualize_det_cv2, img must be a numpy array object.'
    if cls_colors and classes:
        assert len(cls_colors) == len(
            classes), 'cls_colors must be same with classes length if you specific cls_colors.'

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    if blend:
        raw_img = img.copy()
        img = np.zeros_like(img)

    cls_counter = []
    for i in range(detections.shape[0]):
        cls_id = int(detections[i, 0])
        if cls_id != background_id:
            score = detections[i, 1]
            if score > thresh:
                if cls_colors:
                    unique_color = cls_colors[cls_id]
                else:
                    unique_color = get_unique_color_by_id2(cls_id)
                x1, y1, x2, y2 = 0, 0, 0, 0
                if mode == 'xyxy':
                    x1 = int(detections[i, 2])
                    y1 = int(detections[i, 3])
                    x2 = int(detections[i, 4])
                    y2 = int(detections[i, 5])
                else:
                    x1 = int(detections[i, 2])
                    y1 = int(detections[i, 3])
                    x2 = x1 + int(detections[i, 4])
                    y2 = y1 + int(detections[i, 5])

                cv2.rectangle(img, (x1, y1), (x2, y2),
                              unique_color, line_thickness, cv2.LINE_AA)
                if classes:
                    text_label = '{} {:.1f}%'.format(
                        classes[cls_id], score*100)
                    if counter_on:
                        cls_counter.append(classes[cls_id])
                else:
                    text_label = '{} {:.1f}%'.format(cls_id, score*100)

                ((txt_w, txt_h), _) = cv2.getTextSize(
                    text_label, font, font_scale, 1)
                # Place text background.
                back_tl = x1, y1 - int(1.5*txt_h) - 4
                back_br = x1 + 8 + txt_w, y1
                if text_bk:
                    if suit_color:
                        cv2.rectangle(img, back_tl, back_br, unique_color, -1)
                    else:
                        cv2.rectangle(img, back_tl, back_br, (0, 0, 0), -1)
                    # 2 pixel offset from left
                    txt_tl = x1+4, y1 - int(0.5 * txt_h)
                else:
                    txt_tl = x1-txt_w//2 + (x2-x1)//2, y1 - int(0.5 * txt_h)
                if text_bk:
                    cv2.putText(img, text_label, txt_tl, font, font_scale,
                                (0, 0, 0), font_thickness, cv2.LINE_AA)
                else:
                    cv2.putText(img, text_label, txt_tl, font, font_scale,
                                unique_color, font_thickness, cv2.LINE_AA)
    if counter_on:
        cc = Counter(cls_counter)
        cc = OrderedDict(sorted(cc.items()))
        # drw counter result on image
        txt = ''
        for k, v in cc.items():
            txt += '{}: {}\n'.format(k, v)
        put_txt_with_newline(img, txt, counter_pos, font,
                             1.9, (0, 255, 0), 2, cv2.LINE_AA)

    if blend:
        img = cv2.addWeighted(raw_img, 1.0, img, 0.9, 0.6)

    if is_show:
        cv2.imshow('image', img)
        cv2.waitKey(0)
    return img


def visualize_det_cv2_fancy(img, detections, classes=None, thresh=0.2, is_show=False, background_id=-1, mode='xyxy', r=4, d=6):
    """
    visualize detections with a more fancy way

    new add mode option
    mode can be one of 'xyxy' and 'xywh', 'xyxy' as default

    :param img:
    :param detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
    :param classes:
    :param thresh:
    :param is_show:
    :param background_id: -1
    :param mode:
    :return:
    """
    assert isinstance(
        img, np.ndarray), 'from visualize_det_cv2, img must be a numpy array object.'

    height = img.shape[0]
    width = img.shape[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thickness = 1
    line_thickness = 2

    for i in range(detections.shape[0]):
        cls_id = int(detections[i, 0])
        if cls_id != background_id:
            score = detections[i, 1]
            if score > thresh:
                unique_color = create_unique_color_uchar(cls_id)
                x1, y1, x2, y2 = 0, 0, 0, 0
                if mode == 'xyxy':
                    x1 = int(detections[i, 2])
                    y1 = int(detections[i, 3])
                    x2 = int(detections[i, 4])
                    y2 = int(detections[i, 5])
                else:
                    x1 = int(detections[i, 2])
                    y1 = int(detections[i, 3])
                    x2 = x1 + int(detections[i, 4])
                    y2 = y1 + int(detections[i, 5])

                if classes:
                    text_label = '{} {:.1f}%'.format(
                        classes[cls_id], score*100)
                    if counter_on:
                        cls_counter.append(classes[cls_id])
                else:
                    text_label = '{} {:.1f}%'.format(cls_id, score*100)

                _draw_round_dot_border(
                    img, (x1, y1), (x2, y2), unique_color, line_thickness, r, d)
                (txt_size, line_h) = cv2.getTextSize(
                    text_label, font, font_scale, font_thickness)
                txt_org = (int((x1+x2)/2 - txt_size[0]/2), int(y1+line_h+2))
                cv2.putText(img, text_label, txt_org, font, font_scale,
                            (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
    if is_show:
        cv2.imshow('image', img)
        cv2.waitKey(0)
    return img


def visualize_det_cv2_part(img, scores, cls_ids, boxes, class_names=None, thresh=0.2,
                           is_show=False, random=False, background_id=-1, mode='xyxy', style='none',
                           force_color=None, line_thickness=2, font_scale=0.2, wait_t=0):
    """
    visualize detection on image using cv2, this is the standard way to visualize detections

    new add mode option
    mode can be one of 'xyxy' and 'xywh', 'xyxy' as default

    :param img:
    :param detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
    :param classes:
    :param thresh:
    :param is_show:
    :param background_id: -1
    :param mode:
    :return:
    """
    assert isinstance(
        img, np.ndarray), 'from visualize_det_cv2, img must be a numpy array object.'
    if force_color:
        assert isinstance(force_color, list) or isinstance(
            force_color, np.ndarray), 'force_color must be list or numpy array'

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    line_thickness = line_thickness

    n_boxes = 0
    if isinstance(boxes, np.ndarray):
        n_boxes = boxes.shape[0]
    elif isinstance(boxes, list):
        n_boxes = len(boxes)
    else:
        print('boxes with unsupported type, boxes must be ndarray or list.')

    if class_names is None:
        # not using background
        class_names = coco_label_map_list[1:]

    for i in range(n_boxes):
        cls_id = int(cls_ids[i])
        if cls_id != background_id:
            if scores is not None and scores[i] > thresh:
                if force_color:
                    if random:
                        unique_color = force_color[np.random.randint(100)]
                    else:
                        if cls_id > len(force_color)-1:
                            unique_color = force_color[min(
                                cls_id, len(force_color)-1)]
                        else:
                            unique_color = force_color[cls_id]
                else:
                    unique_color = colors(cls_id, True)
                    # unique_color = colors[cls_id]
                x1, y1, x2, y2 = 0, 0, 0, 0
                if mode == 'xyxy':
                    x1 = int(boxes[i, 0])
                    y1 = int(boxes[i, 1])
                    x2 = int(boxes[i, 2])
                    y2 = int(boxes[i, 3])
                else:
                    x1 = int(boxes[i, 0])
                    y1 = int(boxes[i, 1])
                    x2 = x1 + int(boxes[i, 2])
                    y2 = y1 + int(boxes[i, 3])

                if style in ['dashed', 'dotted']:
                    draw_rect_with_style(
                        img, (x1, y1), (x2, y2), unique_color, line_thickness, style=style)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2),
                                  unique_color, line_thickness, cv2.LINE_AA)

                if class_names:
                    if cls_id > len(class_names) - 1:
                        n = class_names[min(cls_id, len(force_color)-1)]
                    else:
                        n = class_names[cls_id]
                    text_label = '{} {:.2f}'.format(n, scores[i])
                else:
                    text_label = '{} {:.2f}'.format(cls_id, scores[i])

                (ret_val, _) = cv2.getTextSize(
                    text_label, font, font_scale, font_thickness)
                txt_bottom_left = (x1+4, y1-4)
                cv2.rectangle(img, (txt_bottom_left[0]-4, txt_bottom_left[1] - ret_val[1]-2),
                              (txt_bottom_left[0] + ret_val[0] +
                               2, txt_bottom_left[1]+4),
                              unique_color, -1, cv2.LINE_AA)
                cv2.putText(img, text_label, txt_bottom_left, font,
                            font_scale, (237, 237, 237), font_thickness, cv2.LINE_AA)
            else:
                if force_color:
                    if random:
                        unique_color = force_color[np.random.randint(100)]
                    else:
                        unique_color = force_color[cls_id]
                else:
                    unique_color = colors(cls_id, True)
                    # unique_color = colors[cls_id]
                x1, y1, x2, y2 = 0, 0, 0, 0
                if mode == 'xyxy':
                    x1 = int(boxes[i, 0])
                    y1 = int(boxes[i, 1])
                    x2 = int(boxes[i, 2])
                    y2 = int(boxes[i, 3])
                else:
                    x1 = int(boxes[i, 0])
                    y1 = int(boxes[i, 1])
                    x2 = x1 + int(boxes[i, 2])
                    y2 = y1 + int(boxes[i, 3])

                if style in ['dashed', 'dotted']:
                    draw_rect_with_style(
                        img, (x1, y1), (x2, y2), unique_color, line_thickness, style=style)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2),
                                  unique_color, line_thickness, cv2.LINE_8)

                if class_names:
                    text_label = '{}'.format(class_names[cls_id])
                else:
                    text_label = '{}'.format(cls_id)

                (ret_val, _) = cv2.getTextSize(
                    text_label, font, font_scale, font_thickness)
                txt_bottom_left = (x1+4, y1-4)
                cv2.rectangle(img, (txt_bottom_left[0]-4, txt_bottom_left[1] - ret_val[1]-2),
                              (txt_bottom_left[0] + ret_val[0] +
                               2, txt_bottom_left[1]+4),
                              unique_color, -1, cv2.LINE_AA)
                cv2.putText(img, text_label, txt_bottom_left, font,
                            font_scale, (237, 237, 237), font_thickness, cv2.LINE_AA)
    if is_show:
        cv2.imshow('image', img)
        cv2.waitKey(wait_t)
    return img


def visualize_det_mask_cv2(img, detections, masks, classes=None, is_show=False, background_id=-1, is_video=False):
    """
    this method using for display detections and masks on image
    :param img:
    :param detections: numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object. contains id and score in the first 2 rows
    :param masks: numpy.array([[mask_width, mask_height], ...], every element is an
    one chanel mask of on object
    :param classes: classes names in a list with certain order
    :param is_show: to show if it is video
    :param background_id
    :param is_video
    :return:
    """
    assert isinstance(img, np.ndarray) and isinstance(detections, np.ndarray) and isinstance(masks, np.ndarray), \
        'images and detections and masks must be numpy array'
    assert detections.shape[0] == masks.shape[-1], 'detections nums and masks nums are not equal'
    assert is_show != is_video, 'you can not set is_show and is_video at the same time.'
    # draw detections first
    img = visualize_det_cv2(img, detections, classes=classes, is_show=False)

    masked_image = img
    print('masked image shape: ', masked_image.shape)
    num_instances = detections.shape[0]
    for i in range(num_instances):
        cls_id = int(detections[i, 0])
        if cls_id != background_id:
            unique_color = create_unique_color_uchar(cls_id)
            mask = masks[:, :, i]
            masked_image = _apply_mask2(masked_image, mask, unique_color)
    # masked_image = masked_image.astype(int)
    if is_video:
        cv2.imshow('image', masked_image)
        cv2.waitKey(1)
    elif is_show:
        cv2.imshow('image', masked_image)
        cv2.waitKey(0)
    return masked_image


def _apply_mask2(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


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
            cv2.line(img, (point_1_[0], point_1_[1]),
                     (point_2_[0], point_2_[1]), color, 1)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i + 2) % 8]
            cv2.line(img, (point_1_[0], point_1_[1]),
                     (point_2_[0], point_2_[1]), color, 1)
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
