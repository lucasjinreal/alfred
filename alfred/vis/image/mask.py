"""
Draw masks on image,
every mask will has single id, and color are not same
also this will give options to draw detection or not


"""
import cv2
import numpy as np
from .common import get_unique_color_by_id
from .det import draw_one_bbox
from PIL import Image


def draw_masks_maskrcnn(image, boxes, scores, labels, masks, human_label_list=None,
                        score_thresh=0.6, draw_box=True):
    """
    Standared mask drawing function

    boxes: a list of boxes, or numpy array
    scores: a list of scores or numpy array
    labels: same as scores
    masks: resize to same width and height as box masks

    NOTE: if masks not same with box, then it will resize inside this function

    TODO: To adding human readable text drawing

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

    # all_masks_empty_image = np.zeros(image.shape, dtype=np.uint8)

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

                    txt = '{} {:.2f}'.format(human_label_list[label], score)
                    cv2.putText(image, txt, (x1, y1), font, font_scale, cls_color, font_thickness)

            # colorize mask
            m_w = int(x2-x1)
            m_h = int(y2-y1)
            mask = Image.fromarray(mask).resize((m_w, m_h), Image.LINEAR)
            mask = np.array(mask)
            # cv2.imshow('rr2', mask)
            # cv2.waitKey(0)

            mask_flatten = mask.flatten()
            # if pixel value less than 0.5, that's background, min: 0.0009, max: 0.9
            mask_flatten_color = np.array(list(map(lambda it: instance_color if it > 0.5 else [0, 0, 0],
                                                   mask_flatten)), dtype=np.uint8)

            mask_color = np.resize(mask_flatten_color, (m_h, m_w, 3))
            # cv2.imshow('rr', mask_color)
            # cv2.waitKey(0)

            empty_image[y1: y2, x1: x2, :] = mask_color
    # combine image and masks
    # now we got mask
    combined = cv2.addWeighted(image, 0.5, empty_image, 0.6, 0)

    # cv2.imshow('rr', empty_image)
    # cv2.imshow('combined', combined)
    # cv2.waitKey(0)
    return combined
