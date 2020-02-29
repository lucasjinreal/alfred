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

count how many certain files under dir

"""
import os
import glob
from alfred.utils.log import logger as logging



def count_file(d, f_type):
    assert os.path.exists(d), '{} not exist.'.format(d)
    # f_type can be jpg,png,pdf etc, connected by comma
    all_types = f_type.split(',')
    logging.info('count all file types: {} under: {}'.format(all_types, d))
    all_files = []
    for t in all_types:
        t = t.replace('.', '')
        one = glob.glob(os.path.join(d, '*.{}'.format(t)))
        one = [i for i in one if os.path.isfile(i)]
        logging.info('{} num: {}'.format(t, len(one)))
        all_files.extend(one)
    logging.info('file types: {}, total num: {}'.format(all_types, len(all_files)))
