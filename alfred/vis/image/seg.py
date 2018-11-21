"""
draw segmentation result

even instance segmentation result

"""
import numpy as np
import cv2
from .get_dataset_colormap import label_to_color_image
from .get_dataset_colormap import _ADE20K, _CITYSCAPES, _MAPILLARY_VISTAS, _PASCAL



def draw_masks(img, masks, cls_color_list, is_show=False, background_id=-1, is_video=False, convert_bgr=False):
    """

    DEPRACATION:

    this method can not work...
    masks are:
    [ [[0, 1, 2,...],[4,...],],
      [[0, 1, 2,...],[4,...],],
    ]
    (n_classes, h, w)

    the color list better using RGBA channel
    cls_color_list = [(223,  224, 225, 0.4), (12, 23, 23, 0.4), ...] a list of colors

    Note: suppose the img in BGR format, you should convert to RGB once img returned
    :param img:
    :param masks:
    :param cls_color_list:
    :param is_show:
    :param background_id:
    :param is_video:
    :return:
    """
    n, h, w = masks.shape

    mask_flatten = masks.flatten()
    mask_color = np.array(list(map(lambda i: cls_color_list[i], mask_flatten)))
    # reshape to normal image shape,
    mask_color = np.reshape(mask_color, (h, w, 3)).astype('float32')

    # add this mask on img
    # img = cv2.add(img, mask_color)
    if convert_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(img, 0.6, mask_color, 0.4, 0)
    if is_show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return img


def draw_seg(img, seg, cls_color_list, alpha=0.6, is_show=False, bgr_in=False):
    """
    NOTE: this method not work well, the add weight method just not work

    Show segmentation result on image.
    the seg is segmentation result which after np.argmax operation

    it's (h, w) size, very pixel value is a class index

    :param img:
    :param seg:
    :param cls_color_list:
    :param is_show:
    :param alpha
    :param bgr_in
    :return:
    """
    h, w = seg.shape
    mask_flatten = seg.flatten()
    mask_color = np.array(list(map(lambda i: cls_color_list[i], mask_flatten)))
    mask_color = np.reshape(mask_color, (h, w, 3)).astype(np.float)

    mask = (seg != 0)
    # convert mask=(h, w) -> mask=(h, w, 3)
    mask = np.dstack((mask, mask, mask)).astype(np.float)
    mask *= alpha

    # out = np.where(mask, mask_color, img)
    out = mask_color * mask + img * (1.0 - mask)
    if is_show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return out, mask_color


def draw_seg_by_dataset(img, seg, dataset, alpha=0.5, is_show=False, bgr_in=False):
    assert dataset in [_PASCAL, _CITYSCAPES, _MAPILLARY_VISTAS, _ADE20K], 'dataset not support yet.'
    img = np.asarray(img, dtype=np.float)
    if bgr_in:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_color = np.asarray(label_to_color_image(seg, dataset), dtype=np.float)

    mask = (seg != 0)
    # convert mask=(h, w) -> mask=(h, w, 3)
    mask = np.dstack((mask, mask, mask)).astype(np.float)
    mask *= alpha

    # out = np.where(mask, mask_color, img)
    out = mask_color * mask + img * (1.0 - mask)
    if is_show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return out, mask_color