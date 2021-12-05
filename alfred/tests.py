# -*- coding: utf-8 -*-
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
from utils.mana import welcome

from utils.log import logger as logging
from vis.image.det import visualize_det_cv2
import cv2
import numpy as np
from vis.image.get_dataset_label_map import coco_label_map_list
from vis.image.common import draw_rect_with_style
import torch
from dl.torch.common import print_tensor

from varname import varname


def a_func(num):
    print(varname() + ': ' + str(num))


def clothes(func):
    def wear():
        print('Buy clothes!{}'.format(func.__name__))
        return func()
    return wear


@clothes
def body():
    print('The body feels could!')


if __name__ == '__main__':
    v = a_func(1098)

    # welcome('')
    # logging.info('hi hiu')
    # logging.error('ops')

    # a = cv2.imread('/home/jintian/Pictures/1.jpeg')

    # dets = [
    #     [1, 0.9, 4, 124, 333, 256],
    #     [2, 0.7, 155, 336, 367, 485],
    # ]
    # dets = np.array(dets)
    # print(type(a))

    # draw_rect_with_style(a, (78, 478), (478, 223), (0, 255, 255), style='dashed')
    # visualize_det_cv2(a, dets, coco_label_map_list, is_show=True)

    aaa = torch.randn([1, 23, 45])
    print_tensor(aaa)
