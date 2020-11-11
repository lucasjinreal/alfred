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
gather all voc labels from Annotations root folder
which contains all xml annotations

"""

"""

gather the label from Annotations
"""
import os
import pickle
import os.path
import sys
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import glob



def gather_labels(anno_dir):
    all_labels = glob.glob(os.path.join(anno_dir, '*.xml'))
    all_names = []
    all_obj_num = 0
    xmls_without_boxes = []
    i = 0
    cls_num_map = dict()
    for label in all_labels:
        if i % 500 == 0:
            print('parsing [{}/{}] {}'.format(i, len(all_labels), label))
        i += 1
        root = ET.parse(label).getroot()
        one_sample_obj_num = 0
        for obj in root.iter('object'):
            one_sample_obj_num += 1
            name = obj.find('name').text
            if name in cls_num_map.keys():
                cls_num_map[name] += 1
            else:
                cls_num_map[name] = 0
            if name not in all_names:
                all_names.append(name)
        if one_sample_obj_num == 0:
            xmls_without_boxes.append(label)
        all_obj_num += one_sample_obj_num
    print('Done. summary...')
    print('all {} classes.'.format(len(all_names)))
    print(all_names)
    # we also read xmls with empty boxes
    print('\nclass boxes statistic as: {}'.format(cls_num_map))
    if len(xmls_without_boxes) > 0:
        print('\nalso, we found these files without any detections, you can consider remove it:')
        print(xmls_without_boxes)


