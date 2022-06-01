#!/usr/bin/python
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

# pip install lxml

import sys
import os
import json
from alfred.utils.log import logger as logging
import glob
import cv2

try:
    import xml.etree.ElementTree as ET
except ImportError:
    pass


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError(
            "Filename %s is supposed to be an integer." % (filename)
        )


"""
xml_list is optional, we at least need xml_dir and json_file
"""


def get_pure_file_name(file_name):
    file_name = os.path.basename(file_name)
    return os.path.splitext(file_name)[0]


def get_file_name_ext(file_name):
    file_name = os.path.basename(file_name)
    return os.path.splitext(file_name)[1]


def convert(xml_dir, img_dir, json_file=None, xml_list=None, index_1=False):
    if index_1:
        print("Annotations save with index start from 1.")
    if xml_list:
        list_fp = open(xml_list, "r")
    else:
        list_fp = glob.glob(os.path.join(xml_dir, "*.xml"))
    logging.info("we got all xml files: {}".format(len(list_fp)))
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID

    i = 0
    for line in list_fp:
        line = line.strip()
        if i % 1000 == 0:
            print("Processing %s" % (line))
        # line could be relative path
        xml_f = line
        if not os.path.exists(line):
            xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise NotImplementedError("%d paths found in %s" % (len(path), line))
        # compare filename with xml filename
        if get_pure_file_name(xml_f) != get_pure_file_name(filename):
            # if not equal, we replace filename with xml file name
            # print('{} != {}'.format(os.path.basename(xml_f).split('.')[0], filename.split('.')[0]))

            filename = get_pure_file_name(xml_f) + get_file_name_ext(filename)

            # filename could be wrong sufix
            print("revise filename to: {}".format(filename))
            if not os.path.exists(os.path.join(os.path.dirname(xml_f), filename)):
                logging.info(
                    "revise filename wrong, try change sufix (but also could be wrong, check your VOC format pls.)"
                )
                filename = get_pure_file_name(filename) + ".jpg"

        # filename to jpg name
        img_filename = get_pure_file_name(filename) + ".jpg"
        if not os.path.exists(os.path.join(img_dir, img_filename)):
            img_filename = get_pure_file_name(filename) + ".png"
        # don't support other format

        # The filename must be a number
        # image_id = get_filename_as_int(filename)
        image_id = i
        try:
            size = get_and_check(root, "size", 1)
            width = int(get_and_check(size, "width", 1).text)
            height = int(get_and_check(size, "height", 1).text)
        except Exception as e:
            # if size not found, we infer from image
            logging.info(
                "{} xml format not fully right, force image height, width, this might not be right, but most cases not effect.".format(
                    xml_f
                )
            )
            height, width, _ = cv2.imread(os.path.join(img_dir, img_filename)).shape

        # check if height or width is normal
        if width <= 0 or height <= 0:
            # we just override width and height stores at xml
            height, width, _ = cv2.imread(os.path.join(img_dir, img_filename)).shape

        image = {
            "file_name": img_filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        # Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories) + 1 if index_1 else len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = float(get_and_check(bndbox, "xmin", 1).text)
            ymin = float(get_and_check(bndbox, "ymin", 1).text)
            xmax = float(get_and_check(bndbox, "xmax", 1).text)
            ymax = float(get_and_check(bndbox, "ymax", 1).text)
            assert xmax >= xmin, "xmax: {} xmin: {} check failed.".format(xmax, xmin)
            assert ymax >= ymin, "ymax: {} ymin: {} check failed.".format(ymax, ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1
        # image_id plus 1
        i += 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)
    if not json_file:
        json_file = "annotations_coco.json"
        logging.info("converted coco format will saved into: {}".format(json_file))
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    logging.info("done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("at least 2 auguments are need.")
        print(
            "Usage: %s XML_LIST.txt(optional) XML_DIR OUTPU_JSON.json" % (sys.argv[0])
        )
        exit(1)
    if len(sys.argv) == 3:
        # xml_dir, output_json
        convert(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        # xml_list, xml_dir, output_json
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
