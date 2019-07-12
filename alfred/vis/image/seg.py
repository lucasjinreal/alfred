"""
draw segmentation result

even instance segmentation result

"""
import numpy as np
import cv2
from .get_dataset_colormap import label_to_color_image
from .get_dataset_colormap import _ADE20K, _CITYSCAPES, _MAPILLARY_VISTAS, _PASCAL


def draw_seg_by_dataset(img, seg, dataset, alpha=0.7, is_show=False, bgr_in=False):
    assert dataset in [_PASCAL, _CITYSCAPES, _MAPILLARY_VISTAS, _ADE20K], 'dataset not support yet.'
    img = np.asarray(img, dtype=np.uint8)
    if bgr_in:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_color = np.asarray(label_to_color_image(seg, dataset), dtype=np.uint8)
    img_shape = img.shape
    mask_shape = mask_color.shape
    if img_shape != mask_shape:
        # resize mask to img shape
        mask_color = cv2.resize(mask_color, (img.shape[1], img.shape[0]))

    res = cv2.addWeighted(img, 0.3, mask_color, alpha, 0.6)
    if is_show:
        cv2.imshow('result', res)
        cv2.waitKey(0)
    return res, mask_color
