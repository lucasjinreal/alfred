# -*- coding: utf-8 -*-
# file: tests.py
# author: JinTian
# time: 16/03/2018 3:23 PM
# Copyright 2018 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
from utils.mana import welcome

from utils.log import logger as logging
from vis.image.det import visualize_det_cv2
import cv2
import numpy as np
from vis.image.get_dataset_label_map import coco_label_map_list


if __name__ == '__main__':
    welcome('')
    logging.info('hi hiu')
    logging.error('ops')

    a = cv2.imread('/home/jintian/Pictures/1.jpeg')

    dets = [
        [1, 0.9, 4, 124, 333, 256],
        [2, 0.7, 155, 336, 367, 485],
    ]
    dets = np.array(dets)
    print(type(a))
    visualize_det_cv2(a, dets, coco_label_map_list, is_show=True)