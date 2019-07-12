#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: alfred.py
# author: JinTian
# time: 04/02/2018 11:59 AM
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
main entrance of Alfred
"""
import os
import sys
import argparse
from colorama import Fore, Back, Style

from .modules.vision.video_extractor import VideoExtractor
from .modules.scrap.image_scraper import ImageScraper
from .modules.vision.to_video import VideoCombiner
from .modules.vision.vis_kit import draw_box_without_score
from .modules.vision.face_extractor import FaceExtractor

__VERSION__ = '2.3'
__AUTHOR__ = 'Lucas Jin'
__DATE__ = '2018.11.11'
__LOC__ = 'Shenzhen, China'
__git__ = 'http://github.com/jinfagang/alfred'


def arg_parse():
    """
    parse arguments
    :return:
    """
    parser = argparse.ArgumentParser(prog="alfred")
    parser.add_argument('--version', '-v', action="store_true", help='show version info.')

    # vision, text, scrap
    main_sub_parser = parser.add_subparsers()

    # =============== vision part ================
    vision_parser = main_sub_parser.add_parser('vision', help='vision related commands.')
    vision_sub_parser = vision_parser.add_subparsers()

    vision_extract_parser = vision_sub_parser.add_parser('extract', help='extract image from video: alfred vision '
                                                                         'extract -v tt.mp4')
    vision_extract_parser.set_defaults(which='vision-extract')
    vision_extract_parser.add_argument('--video', '-v', help='video to extract')
    vision_extract_parser.add_argument('--jumps', '-j', help='jump frames for wide extract')

    vision_2video_parser = vision_sub_parser.add_parser('2video', help='combine into a video: alfred vision '
                                                                       '2video  -d ./images')
    vision_2video_parser.set_defaults(which='vision-2video')
    vision_2video_parser.add_argument('--dir', '-d', help='dir contains image sequences.')

    vision_clean_parser = vision_sub_parser.add_parser('clean', help='clean images in a dir.')
    vision_clean_parser.set_defaults(which='vision-clean')
    vision_clean_parser.add_argument('--dir', '-d', help='dir contains images.')

    vision_getface_parser = vision_sub_parser.add_parser('getface', help='get all faces inside an image and save it.')
    vision_getface_parser.set_defaults(which='vision-getface')
    vision_getface_parser.add_argument('--dir', '-d', help='dir contains images to extract faces.')

    # =============== text part ================
    text_parser = main_sub_parser.add_parser('text', help='text related commands.')
    text_sub_parser = text_parser.add_subparsers()

    text_clean_parser = text_sub_parser.add_parser('clean', help='clean text.')
    text_clean_parser.set_defaults(which='text-clean')
    text_clean_parser.add_argument('--file', '-f', help='file to clean')

    text_translate_parser = text_sub_parser.add_parser('translate', help='translate')
    text_translate_parser.set_defaults(which='text-translate')
    text_translate_parser.add_argument('--file', '-f', help='translate a words to target language')

    # =============== scrap part ================
    scrap_parser = main_sub_parser.add_parser('scrap', help='scrap related commands.')
    scrap_sub_parser = scrap_parser.add_subparsers()

    scrap_image_parser = scrap_sub_parser.add_parser('image', help='scrap images.')
    scrap_image_parser.set_defaults(which='scrap-image')
    scrap_image_parser.add_argument('--query', '-q', help='query words.')

    return parser.parse_args()


def print_welcome_msg():
    print(Fore.BLUE + Style.BRIGHT + 'Alfred ' + Style.RESET_ALL +
          Fore.WHITE + '- Valet of Artificial Intelligence.' + Style.RESET_ALL)
    print('Author: ' + Fore.RED + Style.BRIGHT + __AUTHOR__ + Style.RESET_ALL)
    print('At    : ' + Fore.RED + Style.BRIGHT + __DATE__ + Style.RESET_ALL)
    print('Loc   : ' + Fore.RED + Style.BRIGHT + __LOC__ + Style.RESET_ALL)
    print('Star  : ' + Fore.RED + Style.BRIGHT + __git__ + Style.RESET_ALL)
    print('Ver.  : ' + Fore.RED + Style.BRIGHT + __VERSION__ + Style.RESET_ALL)


def main(args=None):
    args = arg_parse()
    if args.version:
        print(print_welcome_msg())
        exit(0)
    else:
        args_dict = vars(args)
        print_welcome_msg()
        try:
            module = args_dict['which'].split('-')[0]
            action = args_dict['which'].split('-')[1]
            print(Fore.GREEN + Style.BRIGHT)
            print('=> Module: ' + Fore.WHITE + Style.BRIGHT + module + Fore.GREEN + Style.BRIGHT)
            print('=> Action: ' + Fore.WHITE + Style.BRIGHT + action)
            if module == 'vision':
                if action == 'extract':
                    v_f = args_dict['video']
                    j = args_dict['jumps']
                    print(Fore.BLUE + Style.BRIGHT + 'Extracting from {}'.format(v_f))
                    video_extractor = VideoExtractor(jump_frames=j)
                    video_extractor.extract(v_f)
                elif action == '2video':
                    d = args_dict['dir']
                    combiner = VideoCombiner(img_dir=d)
                    print(Fore.BLUE + Style.BRIGHT + 'Combine video from {}'.format(d))
                    print(Fore.BLUE + Style.BRIGHT + 'What the hell.. {}'.format(d))
                    combiner.combine()

                elif action == 'clean':
                    d = args_dict['dir']
                    print(Fore.BLUE + Style.BRIGHT + 'Cleaning from {}'.format(d))

                elif action == 'getface':
                    try:
                        import dlib
                        d = args_dict['dir']
                        print(Fore.BLUE + Style.BRIGHT + 'Extract faces from {}'.format(d))

                        face_extractor = FaceExtractor()
                        face_extractor.get_faces(d)
                    except ImportError:
                        print('This action needs to install dlib first. http://dlib.net')

            elif module == 'text':
                if action == 'clean':
                    f = args_dict['file']
                    print(Fore.BLUE + Style.BRIGHT + 'Cleaning from {}'.format(f))
                elif action == 'translate':
                    f = args.v
                    print(Fore.BLUE + Style.BRIGHT + 'Translate from {}'.format(f))
            elif module == 'scrap':
                if action == 'image':
                    q = args_dict['query']
                    q_list = q.split(',')
                    q_list = [i.replace(' ', '') for i in q_list]
                    image_scraper = ImageScraper()
                    image_scraper.scrap(q_list)

        except Exception as e:
            print(Fore.RED, 'parse args error, type -h to see help. msg: {}'.format(e))


if __name__ == '__main__':
    main()
