# -*- coding: utf-8 -*-
# file: image_scraper.py
# author: JinTian
# time: 05/02/2018 9:37 PM
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
import re
import requests
import os
import argparse
import random
from colorama import Fore, Back, Style


class ImageScraper(object):

    def __init__(self):
        pass

    @staticmethod
    def save_image(image_data, save_prefix, save_dir, index):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if save_prefix:
            if index:
                save_file = os.path.join(save_dir, save_prefix + '_' + str(index) + '.jpg')
            else:
                file_name = ''.join(random.sample('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789', 16))
                save_file = os.path.join(save_dir, save_prefix + '_' + file_name + '.jpg')
        else:
            if index:
                save_file = os.path.join(save_dir, '_' + str(index) + '.jpg')
            else:
                file_name = ''.join(random.sample('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789', 16))
                save_file = os.path.join(save_dir, '_' + file_name + '.jpg')
        with open(save_file, 'wb') as f:
            print('-- image saved into {}'.format(os.path.basename(save_file)))
            f.write(image_data)

    @staticmethod
    def decode_url(url):
        url = url.replace("_z2C$q", ":")
        url = url.replace("_z&e3B", ".")
        url = url.replace("AzdH3F", "/")
        in_table = "wkv1ju2it3hs4g5rq6fp7eo8dn9cm0bla"
        out_table = "abcdefghijklmnopqrstuvw1234567890"
        trans_table = str.maketrans(in_table, out_table)
        url = url.translate(trans_table)
        return url

    def scrap(self, query_words, save_dir=None, max_count=5000):
        print(Fore.BLUE)
        print('-- scrap images of: ' + Fore.YELLOW + Style.BRIGHT + ' '.join(query_words) + Style.RESET_ALL)
        if save_dir:
            root_path = save_dir
        else:
            root_path = os.getcwd()
        for k, query_word in enumerate(query_words):
            url_pattern = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&" \
                          "ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0" \
                          "&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
            urls = (url_pattern.format(word=query_word, pn=p) for p in range(0, max_count, 30))
            for i_u, url in enumerate(urls):
                try:
                    html = requests.get(url).text
                    image_urls = re.findall('"objURL":"(.*?)",', html, re.S)
                    for i, image_url in enumerate(image_urls):
                        try:
                            image_url = self.decode_url(image_url)
                            print('-- decoding url.. : {}'.format(image_url))
                            print('-- solving %d image' % i)
                            image = requests.get(image_url, stream=False, timeout=10).content
                            save_dir = os.path.join(os.path.abspath(root_path), query_words[k])
                            self.save_image(image, query_word.replace(' ', ''), save_dir, str(i_u) + str(i))
                        except requests.exceptions.ConnectionError:
                            print('-- url: %s can not found image.' % image_url)
                            continue
                except Exception as e:
                    print(e)
                    print('-- pass this url.')
                    pass