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

extract VOC image patchs from VOC annotations

"""
"""
read VOC format
annotations and extract image patchs out

"""
import os
import json
import sys
import xml.etree.ElementTree as ET
from alfred.utils.log import logger as logging
import cv2
import argparse


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_dir, output_dir, img_dir, xml_list=None):
    assert os.path.join('image dir {} not exist'.format(img_dir))
    os.makedirs(output_dir, exist_ok=True)
    if xml_list:
        list_fp = open(xml_list, 'r')
    else:
        list_fp = os.listdir(xml_dir)
    logging.info('we got all xml files: {}'.format(len(list_fp)))
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}

    i = 0
    for line in list_fp:
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        # compare filename with xml filename
        if os.path.basename(xml_f).split('.')[0] != filename.split('.')[0]:
            # if not equal, we replace filename with xml file name
            # print('{} != {}'.format(os.path.basename(xml_f).split('.')[0], filename.split('.')[0]))
            filename = os.path.basename(xml_f).split('.')[0] + '.' + filename.split('.')[-1]
            print('revise filename to: {}'.format(filename))
        ## The filename must be a number
        # image_id = get_filename_as_int(filename)
        image_id = i
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        
        img_root = img_dir
        img_f = os.path.join(img_root, os.path.basename(xml_f).replace('xml', 'jpg'))
        assert img_f, '{} not exist'.format(img_f)
        logging.info('reading image from: {}'.format(img_f))
        ori_img = cv2.imread(img_f)
        bx_id = 0
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = float(get_and_check(bndbox, 'xmin', 1).text)
            ymin = float(get_and_check(bndbox, 'ymin', 1).text)
            xmax = float(get_and_check(bndbox, 'xmax', 1).text)
            ymax = float(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            
            # we got an image patch and will save it
            # print('{} - {}, {} - {}'.format(int(xmin), int(xmax), int(ymin), int(ymax)))
            img_patch = ori_img[ int(ymin):int(ymax), int(xmin):int(xmax)]
            # cv2.imshow('rr', img_patch)
            # cv2.waitKey(0)
            # print(category)
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
            to_save_f = os.path.join(output_dir, category, '{}_{}.jpg'.format(i, bx_id))
            cv2.imwrite(to_save_f, img_patch)
            bx_id += 1
        # image_id plus 1
        i += 1
    logging.info('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image patchs from VOC annotations.')
    parser.add_argument('--xml_dir', '-x', type=str, help='xml dir')
    parser.add_argument('--image_dir', '-i', type=str, default='JPEGImages', help='xml dir')
    parser.add_argument('--output_dir', '-o', type=str, default='extracted_out', help='xml dir')
    args = parser.parse_args()

    xml = args.xml_dir
    img = args.image_dir
    output = args.output_dir

    convert(xml, output, img)