# -*- coding: utf-8 -*-
# file: to_video.py
# author: JinTian
# time: 16/03/2018 2:24 PM
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
"""
this part using for combine image sequences into a single video

as previously version, the sequence are not well ordered so that video were not
frequent, we solve that problem now

"""
import os
import cv2
from colorama import Fore, Back, Style
import numpy as np
import sys


class VideoCombiner(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

        if not os.path.exists(self.img_dir):
            print(Fore.RED + '=> Error: ' + '{} not exist.'.format(self.img_dir))
            exit(0)

        self._get_video_shape()

    def _get_video_shape(self):
        self.all_images = [os.path.join(self.img_dir, i) for i in os.listdir(self.img_dir)]
        self.all_images = sorted(self.all_images)
        for item in self.all_images[:int(len(self.all_images) // 2)]:
            print(item)
        # order the images order.

        sample_img = np.random.choice(self.all_images)
        if os.path.exists(sample_img):
            img = cv2.imread(sample_img)
            self.video_shape = img.shape
        else:
            print(Fore.RED + '=> Error: ' + '{} not found or open failed, try again.'.format(sample_img))
            exit(0)

    def combine(self, target_file='combined.mp4'):
        size = (self.video_shape[1], self.video_shape[0])
        print('=> target video frame size: ', size)
        print('=> all {} frames to solve.'.format(len(self.all_images)))
        target_f = 'combined_{}.mp4'.format(os.path.basename(self.img_dir))
        video_writer = cv2.VideoWriter(target_f, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
        i = 0
        print('=> Solving, be patient.')
        for img in self.all_images:
            img = cv2.imread(img, cv2.COLOR_BGR2RGB)
            i += 1
            # print('=> Solving: ', i)
            video_writer.write(img)
        video_writer.release()
        print('Done!')



# d = sys.argv[1]
# combiner = VideoCombiner(d)
# combiner.combine()