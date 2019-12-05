# -*- coding: utf-8 -*-
# file: video_extractor.py
# author: JinTian
# time: 05/02/2018 12:34 PM
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
import os
import sys
import cv2
from colorama import Fore, Back, Style


class VideoExtractor(object):

    def __init__(self, jump_frames=6, save_format='frame_%06d.jpg'):
        """
        we set frames to jump, etc, using jump_frames=6
        will save one frame per 6 frames jumped
        :param jump_frames:
        :param save_format: this is the frames save format
        users can decide what's the format is: frame_0000004.jpg
        """
        self.current_frame = 0
        self.current_save_frame = 0
        if jump_frames:
            self.jump_frames = jump_frames
        else:
            self.jump_frames = 6
        self.save_format = save_format

    def extract(self, video_f):
        if os.path.exists(video_f) and os.path.isfile(video_f):
            cap = cv2.VideoCapture(video_f)

            save_dir = os.path.join(os.path.dirname(video_f), os.path.basename(video_f).split('.')[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            res = True
            while res:
                res, image = cap.read()
                self.current_frame += 1
                if self.current_frame % self.jump_frames == 0:
                    print('Read frame: {} jump frames: {}'.format(self.current_frame, self.jump_frames))
                    cv2.imwrite(os.path.join(save_dir, self.save_format % self.current_save_frame), image)
                    self.current_save_frame += 1

            print(Fore.GREEN + Style.BRIGHT)
            print('Success!')
        else:
            print(Fore.RED + Style.BRIGHT)
            print('Error! ' + Style.RESET_ALL + '{} not exist.'.format(video_f))
