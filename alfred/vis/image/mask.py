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
Draw masks on image,
every mask will has single id, and color are not same
also this will give options to draw detection or not


"""
import torch
import cv2
import numpy as np
from .common import (
    get_unique_color_by_id,
    get_unique_color_by_id2,
    get_unique_color_by_id3,
    get_unique_color_by_id_with_dataset,
)
from .det import draw_one_bbox
from PIL import Image
from .get_dataset_color_map import *
from .get_dataset_label_map import coco_label_map_list

ALL_COLORS_MAP = {
    "cityscapes": create_cityscapes_label_colormap(),
    "mapillary": create_mapillary_vistas_label_colormap(),
    "ade20k": create_ade20k_label_colormap(),
    "voc": create_pascal_label_colormap(),
    "coco": create_coco_stuff_colormap(),
}


def draw_masks_maskrcnn(
    image,
    boxes,
    scores,
    labels,
    masks,
    human_label_list=None,
    score_thresh=0.6,
    draw_box=True,
):
    """
    Standared mask drawing function

    boxes: a list of boxes, or numpy array
    scores: a list of scores or numpy array
    labels: same as scores
    masks: resize to same width and height as box masks

    NOTE: if masks not same with box, then it will resize inside this function

    this function has speed issue be kind of slow.
    and it overlays a box on another mask rather than mask by mask, this is not best
    way.

    :param image:
    :param boxes:
    :param scores:
    :param labels:
    :param masks:
    :param human_label_list
    :param score_thresh
    :param draw_box:
    :return:
    """
    n_instances = 0
    if isinstance(boxes, list):
        n_instances = len(boxes)
    else:
        n_instances = boxes.shape[0]
    # black image with same size as original image
    empty_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(n_instances):
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        mask = masks[i]

        cls_color = get_unique_color_by_id(label)
        # only get RGB
        instance_color = get_unique_color_by_id(i)[:-1]
        # now adding masks to image, and colorize it
        if score >= score_thresh:

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            if draw_box:
                image = draw_one_bbox(image, box, cls_color, 1)
                if human_label_list:
                    # draw text on image
                    font = cv2.QT_FONT_NORMAL
                    font_scale = 0.4
                    font_thickness = 1
                    line_thickness = 1

                    txt = "{} {:.2f}".format(human_label_list[label], score)
                    cv2.putText(
                        image,
                        txt,
                        (x1, y1),
                        font,
                        font_scale,
                        cls_color,
                        font_thickness,
                    )

            # colorize mask
            m_w = int(x2 - x1)
            m_h = int(y2 - y1)
            mask = Image.fromarray(mask).resize((m_w, m_h), Image.LINEAR)
            mask = np.array(mask)
            # cv2.imshow('rr2', mask)
            # cv2.waitKey(0)

            mask_flatten = mask.flatten()
            # if pixel value less than 0.5, that's background, min: 0.0009, max: 0.9
            mask_flatten_color = np.array(
                list(
                    map(
                        lambda it: instance_color if it > 0.5 else [0, 0, 0],
                        mask_flatten,
                    )
                ),
                dtype=np.uint8,
            )

            mask_color = np.resize(mask_flatten_color, (m_h, m_w, 3))
            empty_image[y1:y2, x1:x2, :] = mask_color
    # combine image and masks
    # now we got mask
    combined = cv2.addWeighted(image, 0.5, empty_image, 0.6, 0)
    return combined


def draw_masks_maskrcnn_v2(
    image,
    boxes,
    scores,
    labels,
    masks,
    human_label_list=None,
    score_thresh=0.6,
    draw_box=True,
):
    """
    We change way to draw masks on image

    :param image:
    :param boxes:
    :param scores:
    :param labels:
    :param masks:
    :param human_label_list
    :param score_thresh
    :param draw_box:
    :return:
    """
    n_instances = 0
    if isinstance(boxes, list):
        n_instances = len(boxes)
    else:
        n_instances = boxes.shape[0]
    # black image with same size as original image
    empty_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(n_instances):
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        mask = masks[i]

        cls_color = get_unique_color_by_id(label)
        # only get RGB
        instance_color = get_unique_color_by_id(i)[:-1]
        # now adding masks to image, and colorize it
        if score >= score_thresh:

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            if draw_box:
                image = draw_one_bbox(image, box, cls_color, 1)
                if human_label_list:
                    # draw text on image
                    font = cv2.QT_FONT_NORMAL
                    font_scale = 0.4
                    font_thickness = 1
                    line_thickness = 1

                    txt = "{} {:.2f}".format(human_label_list[label], score)
                    cv2.putText(
                        image,
                        txt,
                        (x1, y1),
                        font,
                        font_scale,
                        cls_color,
                        font_thickness,
                    )

            # colorize mask
            m_w = int(x2 - x1)
            m_h = int(y2 - y1)
            mask = Image.fromarray(mask).resize((m_w, m_h), Image.LINEAR)
            mask = np.array(mask)
            mask_flatten = mask.flatten()
            # if pixel value less than 0.5, that's background, min: 0.0009, max: 0.9
            mask_flatten_color = np.array(
                list(
                    map(
                        lambda it: instance_color if it > 0.5 else [0, 0, 0],
                        mask_flatten,
                    )
                ),
                dtype=np.uint8,
            )

            mask_color = np.resize(mask_flatten_color, (m_h, m_w, 3))
            empty_image[y1:y2, x1:x2, :] = mask_color
    # combine image and masks
    # now we got mask
    combined = cv2.addWeighted(image, 0.5, empty_image, 0.6, 0)
    return combined


# more fast mask drawing here
def vis_bitmasks(
    img,
    bitmasks,
    classes=None,
    fill_mask=True,
    return_combined=True,
    thickness=1,
    draw_contours=True,
):
    """
    visualize bitmasks on image
    """
    # need check if img and bitmask with same W,H
    if isinstance(bitmasks, torch.Tensor):
        bitmasks = bitmasks.cpu().numpy()

    font = cv2.QT_FONT_NORMAL
    font_scale = 0.4
    font_thickness = 1
    res_m = np.zeros_like(img).astype(np.uint8)
    assert isinstance(bitmasks, np.ndarray), "bitmasks must be numpy array"
    bitmasks = bitmasks.astype(np.uint8)
    for i, m in enumerate(bitmasks):
        if m.shape != img.shape:
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
        cts, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # inssue this is a unique color
        if classes is not None:
            c = get_unique_color_by_id2(classes[i])
        else:
            c = get_unique_color_by_id2(i)
        if return_combined:
            if fill_mask:
                cv2.drawContours(
                    res_m, cts, -1, color=c, thickness=-1, lineType=cv2.LINE_AA
                )
                if draw_contours:
                    cv2.drawContours(
                        img, cts, -1, color=c, thickness=thickness, lineType=cv2.LINE_AA
                    )
            else:
                cv2.drawContours(
                    res_m, cts, -1, color=c, thickness=thickness, lineType=cv2.LINE_AA
                )
        else:
            if fill_mask:
                cv2.drawContours(
                    img, cts, -1, color=c, thickness=-1, lineType=cv2.LINE_AA
                )
            else:
                cv2.drawContours(
                    img, cts, -1, color=c, thickness=thickness, lineType=cv2.LINE_AA
                )
    if return_combined:
        img = cv2.addWeighted(img, 0.6, res_m, 0.7, 0.8)
        return img
    else:
        return img


def vis_bitmasks_with_classes(img, classes, bitmasks, force_colors=None, scores=None, class_names=None, mask_border_color=None, draw_contours=False, alpha=0.8, fill_mask=True, return_combined=True, thickness=2):
    """
    visualize bitmasks on image
    """
    # need check if img and bitmask with same W,H
    if isinstance(bitmasks, torch.Tensor):
        bitmasks = bitmasks.cpu().numpy()

    if class_names is None or len(class_names) == 0:
        class_names = coco_label_map_list[1:]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    res_m = np.zeros_like(img)
    assert isinstance(bitmasks, np.ndarray), "bitmasks must be numpy array"
    bitmasks = bitmasks.astype(np.uint8)
    for i, m in enumerate(bitmasks):
        # if m.shape != img.shape:
        #     m = cv2.resize(m, (img.shape[1], img.shape[0]))
        cts, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(cts) > 0:
            cts = max(cts, key=cv2.contourArea)
        # enssue this is a unique color
        cid = int(classes[i])
        if force_colors:
            c = force_colors[cid]
        else:
            c = get_unique_color_by_id3(cid)
        if len(cts) > 0:
            if return_combined:
                if fill_mask:
                    cv2.drawContours(res_m, [cts], -1,  color=c,
                                    thickness=-1, lineType=cv2.LINE_AA)
                    if draw_contours:
                        if mask_border_color:
                            c = mask_border_color
                        cv2.drawContours(img, [cts], -1,  color=c,
                                        thickness=thickness, lineType=cv2.LINE_AA)
                else:
                    cv2.drawContours(res_m, [cts], -1,  color=c,
                                    thickness=thickness, lineType=cv2.LINE_AA)
            else:
                if fill_mask:
                    cv2.drawContours(img, [cts], -1,  color=c,
                                    thickness=-1, lineType=cv2.LINE_AA)
                else:
                    cv2.drawContours(img, [cts], -1,  color=c,
                                    thickness=thickness, lineType=cv2.LINE_AA)
        if classes is not None:
            txt = f"{class_names[classes[i]]}"
            if scores is not None:
                txt += f' {scores[classes[i]]}'
            if len(cts) > 0:
                M = cv2.moments(cts)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # draw labels 
                cv2.putText(img, txt, (cx, cy), font, font_scale, [255, 255, 255], 1, cv2.LINE_AA)
    if return_combined:
        img = cv2.addWeighted(img, 0.9, res_m, alpha, 0.8)
        return img
    else:
        return img


# helper functions
def label2color_mask(
    cls_id_mask, max_classes=90, override_id_clr_map=None, color_suit="cityscapes"
):
    """
    cls_id_mask is your segmentation output
    override_id_clr_map: {2: [0, 0, 0]} used to override color
    """
    assert color_suit in ALL_COLORS_MAP.keys(), "avaiable keys: {}".format(
        ALL_COLORS_MAP.keys()
    )
    colors = ALL_COLORS_MAP[color_suit]
    if override_id_clr_map != None:
        if isinstance(override_id_clr_map, dict):
            colors = np.array(
                [
                    get_unique_color_by_id_with_dataset(i)
                    if i not in override_id_clr_map.keys()
                    else override_id_clr_map[i]
                    for i in range(max_classes)
                ]
            )
        else:
            colors = np.array(
                [
                    override_id_clr_map[i % len(override_id_clr_map)]
                    for i in range(max_classes)
                ]
            )
    if len(colors) < max_classes:
        colors = np.append(
            colors, ALL_COLORS_MAP["cityscapes"][: max_classes - len(colors)], axis=0
        )

    s = cls_id_mask.shape
    if len(s) > 1:
        cls_id_mask = cls_id_mask.flatten()
    mask = colors[cls_id_mask]
    mask = np.reshape(mask, (*s, 3)).astype(np.uint8)
    return mask
