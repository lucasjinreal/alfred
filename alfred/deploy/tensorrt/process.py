# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
import torchvision.transforms as T

import pycuda.driver as cuda
import pycuda.autoinit

import cupy as cp
# from cupy.core.dlpack import toDlpack
# from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from alfred.dl.torch.common import device


def preprocess_np(img_path):
    '''process use numpy
    '''
    im = Image.open(img_path)
    img = im.resize((800, 800),Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2,0,1)
    # print(img.shape)
    img = (img - np.array([ [[0.485]], [[0.456]], [[0.406]] ]))/np.array([ [[0.229]], [[0.224]], [[0.225]] ])

    # img = img.transpose(1,2,0)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = np.array(img).astype(np.float32)

    return img, im, im.size


class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    '''代码来源:
        https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/python/app_onnx_resnet50.py
    '''
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor
    def get_pointer(self):
        return self.tensor.data_ptr()

transform = T.Compose([
    T.Resize((800,800)),  # PIL.Image.BILINEAR
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_torch(img_path):
    '''process use torchvision
    '''
    im = Image.open(img_path)
    img = transform(im).unsqueeze(0)
    img = PyTorchTensorHolder(img)
    return img, im, im.size

def preprocess_torch_v1(img_path):
    im = Image.open(img_path)
    img = transform(im).unsqueeze(0).cpu().numpy()
    return img, im, im.size

def preprocess_np_no_normalize(img_path):
    im = cv2.imread(img_path)
    print(img_path)
    print(im.shape)
    # img = transform(im).unsqueeze(0)
    a = np.transpose(im, (2, 0, 1)).astype(np.float32)
    return a, im


def preprocess_cu(img_np):
    mean_cp = cp.array([ [[0.485]], [[0.456]], [[0.406]] ])
    std_cp = cp.array([ [[0.229]], [[0.224]], [[0.225]] ])

    img_cu = cp.divide(cp.asarray(img,dtype=cp.float32),255.0)
    img_cu = img_cu.transpose(2,0,1)
    img_cu = cp.subtract(img_cu,mean_cp)
    img_cu = cp.divide(img_cu,std_cp)

    # cupy to torch tensor
    # img_tensor = from_dlpack(toDlpack(img_cu))

    return img_cu