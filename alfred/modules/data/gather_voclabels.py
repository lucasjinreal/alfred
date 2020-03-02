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
    for label in all_labels:
        print('parsing {}'.format(label))
        root = ET.parse(label).getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name not in all_names:
                all_names.append(name)
    print('Done. summary...')
    print('all {} classes.'.format(len(all_names)))
    print(all_names)


