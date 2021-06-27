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
common functionality in visualization kit
"""
import numpy as np
import cv2
import colorsys
from itertools import cycle

from .constants import light_colors, dark_colors
from .get_dataset_color_map import create_cityscapes_label_colormap


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def create_unique_color_float(tag, hue_step=0.41, alpha=0.7):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b, alpha


def create_unique_color_uchar(tag, hue_step=0.41, alpha=0.7):
    r, g, b, a = create_unique_color_float(tag, hue_step, alpha)
    return int(255 * r), int(255 * g), int(255 * b), int(255 * a)


def get_unique_color_by_id(idx, alpha=0.7):
    """
    this method can be using when get unique color from id
    or something else
    :param idx:
    :param alpha:
    :return:
    """
    return create_unique_color_uchar(idx, alpha)


def get_unique_color_by_id2(idx, dark=False):
    if dark:
        idx = idx % len(dark_colors)
        return dark_colors[idx]
    else:
        idx = idx % len(light_colors)
        return light_colors[idx]


def get_unique_color_by_id_with_dataset(idx):
    colors = create_cityscapes_label_colormap()
    idx = idx % len(colors)
    return colors[idx]


"""
we need some help functions to draw doted rectangle in opencv
"""


def _drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y = int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    elif style == 'dashed':
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1
    else:
        ValueError('style can only be dotted or dashed for now!')


def _drawpoly(img, pts, color, thickness=1, style='dotted',):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        _drawline(img, s, e, color, thickness, style, gap=6)


def draw_rect_with_style(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    _drawpoly(img, pts, color, thickness, style)
    return img


def put_txt_with_newline(image, multi_line_txt, pt, font, font_scale, color, thickness, line_type):
    text_size, _ = cv2.getTextSize(multi_line_txt, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y0 = pt
    for i, line in enumerate(multi_line_txt.split("\n")):
        y = y0 + i * line_height
        cv2.putText(image,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    line_type)


if __name__ == '__main__':
    c = create_unique_color_uchar(1)
    print(c)
