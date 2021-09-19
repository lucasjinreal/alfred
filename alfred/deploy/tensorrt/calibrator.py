#
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
#

import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes



class Calibrator(trt.IInt8EntropyCalibrator2):
    '''calibrator
        IInt8EntropyCalibrator2
        IInt8LegacyCalibrator
        IInt8EntropyCalibrator
        IInt8MinMaxCalibrator

    '''
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)       
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        # print(self.cache_file)
        stream.reset()
        

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):

        batch = self.stream.next_batch()
        if not batch.size:  
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print(f"[INFO] Using calibration cache to save time: {self.cache_file}")
                return f.read()

    def write_calibration_cache(self, cache): 
        with open(self.cache_file, "wb") as f:
            print(f"[INFO] Caching calibration data for future use: {self.cache_file}")
            f.write(cache)
