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

this tool helps viewing VOC format data


"""
import os
import sys
import cv2
import xml.etree.ElementTree as ET
from glob import glob
import os
import sys
import cv2
from alfred.utils.log import logger as logging
from pascal_voc_writer import Writer


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get(root, name):
    vars = root.findall(name)
    return vars

def merge_multiple_voc_labels(n_merge_files, target_save_root):
    # one of n_merge_files must exist, but no sure which one, found it out
    f_name = os.path.basename(n_merge_files[0])
    n = len(n_merge_files)
    for f in n_merge_files:
        if os.path.exists(f):
            tree = ET.parse(f)
            root = tree.getroot()
            objs = get(root, 'object')
            n_merge_files.remove(f)
            break
    for xml in n_merge_files:
        if os.path.exists(xml):
            t = ET.parse(xml)
            r = t.getroot()
            for obj in get(r, 'object'):
                root.append(obj)
    # how to save 

    # f = open(os.path.join(target_save_root, f_name), 'wb')
    # f.write(etree.tostring(root, pretty_print=True))
    # f.close()
    tree.write(os.path.join(target_save_root, f_name))
    print('merged {} xmls and saved into: {}'.format(n, os.path.join(target_save_root, f_name)))


def merge_voc(label_root_list, style='intersection', label_major=True):
    """
    For intersection: only merges their intersection part;
    For union: will merge all of them

    merge VOC with multiple datasets (one of them may have partial object labeled)
    """
    logging.info('labels root: {}'.format(label_root_list))
    # these labels may not perfectly aligned, we get a union of them
    filenames = []
    if style == 'union':
        logging.info('merge with union style.')
        for l in label_root_list:
            xmls = [os.path.basename(i) for i in glob(os.path.join(l, "*.xml"))]
            logging.info('found {} xmls under: {}'.format(len(xmls), l))
            # filenames.extend([i for i in xmls if i not in filenames])
            filenames.extend(xmls)
    else:
        logging.info('merge with intersection style.')
        for l in label_root_list:
            xmls = [os.path.basename(i) for i in glob(os.path.join(l, "*.xml"))]
            logging.info('found {} xmls under: {}'.format(len(xmls), l))
            # filenames.extend([i for i in xmls if i not in filenames])
            if len(filenames) > 0:
                filenames = set(xmls) & set(filenames)
            filenames = xmls

    filenames = list(set(filenames))
    logging.info('found {} xmls which exist both inside all provided label roots.'.format(len(filenames)))
    target_save_root = './merged_voc_annotations'
    os.makedirs(target_save_root, exist_ok=True)
    for f in filenames:
        n_merge_files = []
        for l in label_root_list:
            n_merge_files.append(os.path.join(l, f))
        merge_multiple_voc_labels(n_merge_files, target_save_root)
    print('done.')


if __name__ == "__main__":
    merge_voc(sys.argv[1], sys.argv[2])
