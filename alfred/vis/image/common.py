"""
common functionality in visualization kit
"""
import numpy as np
import cv2
import colorsys


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