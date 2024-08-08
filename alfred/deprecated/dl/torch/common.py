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
import itertools
import inspect
from json.tool import main
from colorama import Fore, Back, Style
import numpy as np

try:
    import torch

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
except ImportError:
    device = None
    torch_installed = False


def print_tensor(t, label=None, ignore_value=True):
    if isinstance(t, torch.Tensor):
        if label:
            print(Fore.YELLOW + Style.BRIGHT + "-> {}".format(label) + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + Style.BRIGHT + "tensor: " + Style.RESET_ALL)
        if ignore_value:
            print("shape: {}\ndtype: {} {}\n".format(t.shape, t.dtype, t.device))
        else:
            print("value: {}\nshape: {}\ndtype: {}\n".format(t, t.shape, t.dtype))

    else:
        print("{} is not a tensor.".format(t))


def decorator(f):
    def wrapper(*args, **kwargs):
        bound_args = inspect.signature(f).bind(*args, **kwargs)
        bound_args.apply_defaults()

        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args_ori_names = string[string.find("(") + 1 : -1].split(",")

        names = []
        for i in args_ori_names:
            if i.find("=") != -1:
                names.append(i.split("=")[1].strip())
            else:
                names.append(i)
        args_dict = dict(zip(names, args))
        for k, v in args_dict.items():
            k = k.strip()
            if isinstance(v, torch.Tensor):
                print(f"[{k}]: ", v.shape, v.device, v.dtype)
            else:
                print(f"[{k}]: ", v.shape)
        return f(*args, **kwargs)

    return wrapper


@decorator
def print_shape(*vs):
    pass


if __name__ == "__main__":
    cam = torch.randn([4, 5, 300])
    pose = torch.randn([1, 44, 55])
    # print_shape(locals(), cam, pose)
    print_shape(cam, pose)
