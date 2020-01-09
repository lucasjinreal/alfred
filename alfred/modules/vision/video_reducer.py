# -*- coding: utf-8 -*-
# file: video_extractor.py
# author: JinTian
# time: 05/02/2020 12:34 PM
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
from alfred.utils.log import logger as logging


class VideoReducer(object):

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

    def act(self, video_f):
        """
        reduce the video frame by drop frames 
        
        """
        if os.path.exists(video_f) and os.path.isfile(video_f):
            cap = cv2.VideoCapture(video_f)
            target_f = os.path.join(os.path.dirname(video_f), os.path.basename(video_f) + '_reduced.mp4')
            size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(target_f, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
            res = True
            while res:
                res, image = cap.read()
                self.current_frame += 1
                if self.current_frame % self.jump_frames == 0:
                    print('Read frame: {} jump frames: {}'.format(self.current_frame, self.jump_frames))
                    cv2.imwrite(os.path.join(save_dir, self.save_format % self.current_save_frame), image)
                    self.current_save_frame += 1
                    video_writer.write(image)
            video_writer.close()
            logging.info('reduced video file has been saved into: {}'.format(target_f))
        else:
            print(Fore.RED + Style.BRIGHT)
            print('Error! ' + Style.RESET_ALL + '{} not exist.'.format(video_f))
