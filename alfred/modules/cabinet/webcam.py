#
# Copyright (c) 2021 JinTian.
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

showing webcam if you have USB camera
this is using for test your USB usable or not
"""
import cv2 as cv
import os
from colorama import Fore, Back, Style


def webcam_test(vf):
    if vf is not None and os.path.isfile(vf):
        print(Fore.CYAN + 'webcam on: ',
              Style.RESET_ALL, vf, ' press q to quit.')
        cap = cv.VideoCapture(vf)
        while cap.isOpend():
            ret, frame = cap.read()

            if not ret:
                break

            data = preprocess(frame)
            cmap, paf = model(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            # , cmap_threshold=0.15, link_threshold=0.15)
            counts, objects, peaks = parse_objects(cmap, paf)
            draw_objects(frame, counts, objects, peaks)

            cv.imshow('res', frame)
            cv.waitKey(1)
    else:
        print(Fore.CYAN + 'test webcam, press q to quit.', Style.RESET_ALL)
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv.imshow('Webcam', frame)
            if cv.waitKey(1) == ord('q'):
                break
