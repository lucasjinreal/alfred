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
Common utility of pytorch

this contains code that frequently used while writing torch applications

"""
import torch
from colorama import Fore, Back, Style


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def print_tensor(t, label=None, ignore_value=True):
    if isinstance(t, torch.Tensor):
        if label:
            print(Fore.YELLOW + Style.BRIGHT + "-> {}".format(label) + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + Style.BRIGHT + "tensor: " + Style.RESET_ALL)
        if ignore_value:
            print('shape: {}\ndtype: {} {}\n'.format(t.shape, t.dtype, t.device))
        else:
            print('value: {}\nshape: {}\ndtype: {}\n'.format(
            t, t.shape, t.dtype
        ))
    
    else:
        print('{} is not a tensor.'.format(t))