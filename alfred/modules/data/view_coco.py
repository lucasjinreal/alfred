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

this script will using pycoco API
draw our converted annotation to check
if result is right or not

"""
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except ImportError as e:
    print('Got import error: {}'.format(e))
    print('[WARN] you are not either install pycocotools or its dependencies. pls install first.')
    # exit(-1)
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from alfred.utils.log import logger as logging
import cv2
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.common import get_unique_color_by_id
import numpy as np
from pprint import pprint


# USED_CATEGORIES_IDS = [i for i in range(1, 16)]

def get_random_color():
    return list(np.array(np.random.random(size=3) * 256).astype(np.uint8))


def showAnns(ori_img, anns, draw_bbox=False):
    h, w, c = ori_img.shape
    if len(anns) == 0:
        return ori_img
    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'
    elif 'caption' in anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        mask = np.zeros_like(ori_img).astype(np.uint8)

        for ann in anns:
            c = np.array((np.random.random((1, 3)) * 0.6 +
                          0.4)[0]*255).astype(int).tolist()
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(
                            (int(len(seg) / 2), 2))
                        pts = poly.reshape((-1, 1, 2))
                        cv2.polylines(
                            ori_img, np.int32([pts]), True, c, thickness=1, lineType=cv2.LINE_AA)
                        cv2.drawContours(mask, np.int32([pts]), -1, c, -1)

                        if cv2.contourArea(np.int32(pts)) > 1:
                            M = cv2.moments(np.int32(pts))
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(ori_img, 'CAT:{}'.format(
                                ann['category_id']), (cX, cY), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # mask
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']],
                                                    h, w)
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)
                    img = np.ones((m.shape[0], m.shape[1], 3)).astype(np.uint8)
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0, 166.0, 101.0])
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    img *= get_random_color()
                    img = cv2.bitwise_or(img, img, mask=m)
                    mask += img
            if draw_bbox:
                if 'bbox' in ann.keys():
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    pt1 = (int(bbox_x), int(bbox_y))
                    pt2 = (int(bbox_x+bbox_w), int(bbox_y+bbox_h))
                    cv2.rectangle(ori_img, pt1, pt2, color=c,
                                  thickness=1, lineType=cv2.LINE_AA)

            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                # sks = np.array(
                #     self.loadCats(ann['category_id'])[0]['skeleton']) - 1
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                # for sk in sks:
                #     if np.all(v[sk] > 0):
                #         cv2.line(ori_img, x[sk], y[sk], color=c)
                print(kp)
                print('keypoint vis not supported')

        if type(ann['segmentation']) == list:
            ori_img = cv2.addWeighted(ori_img, 0.7, mask, 0.6, 0.7)
        else:
            print('[WARN] you are using RLE mask encode format.')
            ori_img = cv2.addWeighted(ori_img, 0.7, mask, 0.6, 0.7)
    elif datasetType == 'captions':
        for ann in anns:
            print(ann['caption'])
    return ori_img


def vis_coco(coco_img_root, ann_f):
    data_dir = coco_img_root
    coco = COCO(ann_f)

    cats = coco.loadCats(coco.getCatIds())
    logging.info('cats: {}'.format(cats))
    cats_new = dict()
    for c in cats:
        cats_new[c['id']] = c['name']
    pprint(cats_new)
    pprint([i['name'] for i in cats])
    print('All {} classes.'.format(len(cats_new)))
    img_ids = coco.getImgIds()
    logging.info('all images we got: {}'.format(len(img_ids)))

    # draw instances
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        print('checking img: {}, id: {}'.format(img, img_id))

        # img['file_name'] may be not basename
        if 'file_name' in img.keys():
            img_f = os.path.join(data_dir, os.path.basename(img['file_name']))
        elif 'filename' in img.keys():
            img_f = os.path.join(data_dir, os.path.basename(img['filename']))
        else:
            print(
                'does not foud a file_name or filename in feild. check your annotation style: ', img)
        assert(os.path.exists(
            img_f)), '{} not found, maybe your filename pattern not right? Pls fire an issue to alfred github repo!'.format(img_f)
        anno_ids = coco.getAnnIds(imgIds=img['id'])
        annos = coco.loadAnns(anno_ids)

        logging.info('showing anno: {} objects. '.format(len(annos)))
        if len(annos) > 0 and len(annos[0]['segmentation']) == 0:
            logging.info('no segmentation found, using opencv vis.')
            img = cv2.imread(img_f)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.36
            font_thickness = 1
            line_thickness = 1

            for ann in annos:
                b = ann['bbox']
                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int(x1 + b[2])
                y2 = int(y1 + b[3])
                cls_id = ann['category_id']
                unique_color = get_unique_color_by_id(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2),
                              unique_color, line_thickness, cv2.LINE_AA)
                if cls_id > len(cats_new):
                    print('WARN: seems your category id not same with your meta info!!')
                text_label = '{}:{}'.format(cls_id, cats_new[cls_id])
                (ret_val, _) = cv2.getTextSize(
                    text_label, font, font_scale, font_thickness)
                txt_bottom_left = (x1+4, y1-4)
                cv2.rectangle(img, (txt_bottom_left[0]-4, txt_bottom_left[1] - ret_val[1]-2),
                              (txt_bottom_left[0] + ret_val[0] +
                               2, txt_bottom_left[1]+4),
                              (0, 0, 0), -1)
                cv2.putText(img, text_label, txt_bottom_left, font,
                            font_scale, (237, 237, 237), font_thickness, cv2.LINE_AA)
            cv2.imshow('rr', img)
            cv2.waitKey(0)
        else:
            im = cv2.imread(img_f)
            # plt.imshow(I)
            # plt.axis('off')
            # coco.showAnns(annos, True)
            # plt.show()
            ori_im = showAnns(im, annos, True)
            if ori_im is not None:
                cv2.imshow('aa', ori_im)
                cv2.waitKey(0)
            else:
                I = Image.open(img_f)
                plt.imshow(I)
                plt.axis('off')
                coco.showAnns(annos, True)
                plt.show()
