"""


copy and paste coco annotation
to yolo

"""

import os
import sys
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except ImportError as e:
    print('[WARN] coco2yolo need pycocotools installed.')
    # exit(-1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from alfred.utils.log import logger as logging
import cv2
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.common import get_unique_color_by_id
import shutil


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def coco2yolo(img_r, j_f):
    data_dir = img_r
    coco = COCO(j_f)

    cats = coco.loadCats(coco.getCatIds())
    logging.info('cats: {}'.format(cats))
    print('cls list for yolo\n')
    for i in range(len(cats)):
        print(cats[i]['name'])
    print('\n')
    print('all {} categories.'.format(len(cats)))

    img_ids = coco.getImgIds()

    target_txt_r = os.path.join(os.path.dirname(img_r), 'yolo', 'labels')
    target_img_r = os.path.join(os.path.dirname(img_r), 'yolo', 'images')
    os.makedirs(target_txt_r, exist_ok=True)
    os.makedirs(target_img_r, exist_ok=True)

    print('solving, this gonna take some while...')
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        # print('checking img: {}, id: {}'.format(img, img_id))
        # img['file_name'] may be not basename
        img_f = os.path.join(data_dir, os.path.basename(img['file_name']))
        if not os.path.exists(img_f):
            # if not then pull it back to normal mode
            img_f = os.path.join(data_dir, img['file_name'])
        anno_ids = coco.getAnnIds(imgIds=img['id'])
        annos = coco.loadAnns(anno_ids)

        out_file = open(os.path.join(target_txt_r, os.path.basename(img_f).split('.')[0] + '.txt'), 'w')
        img = cv2.imread(img_f)
        h, w, _ = img.shape
        shutil.copy(img_f, os.path.join(target_img_r, os.path.basename(img_f)))
        for ann in annos:
            b = ann['bbox']
            x1 = int(b[0])
            y1 = int(b[1])
            x2 = int(x1 + b[2])
            y2 = int(y1 + b[3])
            cls_id = ann['category_id']
            b = [x1, x2, y1, y2]
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                    " ".join([str(a) for a in bb]) + '\n')
        out_file.close()
    print('convert to yolo done!')
        