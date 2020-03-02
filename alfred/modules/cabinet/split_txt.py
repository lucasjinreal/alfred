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

split txt file with ratios

alfred cab split -f all.txt -r 0.1,0.8,0.1 -n train,val,test

"""
import os
import glob
from alfred.utils.log import logger as logging
import numpy as np


def split_txt_file(f, ratios, names):
    assert os.path.exists(f), '{} not exist.'.format(f)
    if not ratios:
        ratios = [0.2, 0.8]
    else:
        ratios = [float(i) for i in ratios.split(',')]
    logging.info('split ratios: {}'.format(ratios))

    if not names:
        names = ['part_{}'.format(i) for i in range(len(ratios))]
    else:
        names = names.split(',')
    names = [i+'.txt' for i in names]
    logging.info('split save to names: {}'.format(names))

    a = sum(ratios)
    if a != 1.:
        logging.info(
            'ratios: {} does not sum to 1. you must change it first.'.format(ratios))
        exit(1)

    # read txt file
    with open(f, 'r') as f:
        lines = f.readlines()
        lines_no_empty = [i for i in lines if i != '' and i != '\n']
        logging.info('to split file have all {} lines. droped {} empty lines.'.format(len(lines),
                                                                                      len(lines) - len(lines_no_empty)))
        lines = lines_no_empty
        # split with ratios
        last_lines = 0
        for i, r in enumerate(ratios):
            one = lines[last_lines: last_lines+int(r * len(lines))]
            with open(names[i], 'w') as ff:
                ff.writelines(one)
                logging.info('Part {} saved into: {}. portion: {}/{}={}'.format(
                    i, names[i], len(one), len(lines), len(one)/(len(lines))))
            last_lines += len(one)
    logging.info('split done.')
