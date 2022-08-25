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

from typing import OrderedDict
import numpy as np
import tensorrt as trt
from .calibrator import Calibrator
from alfred.utils.log import logger

import sys
import os
import time
import os
import os.path as osp
import ctypes

try:
    import pycuda.driver as cuda
    # https://documen.tician.de/pycuda/driver.html
    import pycuda.autoinit
except ImportError as e:
    print(
        f'pycuda not installed, or import failed. inference on trt will be disabled. error: {e}')

TRT8 = 8
TRT7 = 7

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
# TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
# Allocate host and device buffers, and create a stream.


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # <--------- the main diff to v2
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def allocate_buffers_v2(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = abs(trt.volume(engine.get_binding_shape(binding))) * \
            engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
    # return inputs, outputs, bindings


# do inference  multi outputs
def do_inference_v2(context, bindings, inputs, outputs, stream, input_tensor):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def allocate_buffers_v2_dynamic(engine, is_explicit_batch=False, input_shape=None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        if dims[-1] == -1:
            assert (
                input_shape is not None
            ), "dynamic trt engine must specific input_shape"
            dims[-2], dims[-1] = input_shape
        size = abs(trt.volume(dims)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def allocate_buffers_v2_dynamic_batch(engine, is_explicit_batch=False, max_batch=1):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        if dims[-1] == -1:
            assert (
                input_shape is not None
            ), "dynamic trt engine must specific input_shape"
            dims[-2], dims[-1] = input_shape
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2_dynamic(context, bindings, inputs, outputs, stream, input_tensor):
    """
    this works for infer on dynamic engine, such as input dynamic
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def build_engine_onnx(
    model_file,
    engine_file,
    FP16=False,
    verbose=False,
    dynamic_input=False,
    batch_size=1,
    chw_shape=None,
):
    def get_engine():
        EXPLICIT_BATCH = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network,builder.create_builder_config() as config, trt.OnnxParser(network,TRT_LOGGER) as parser:
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:

            trt_version = int(trt.__version__[0])
            # Workspace size is the maximum amount of memory available to the builder while building an engine.

            if trt_version == TRT8:
                config.max_workspace_size = 6 << 30  # 2GB
            else:
                builder.max_workspace_size = 6 << 30  # 2GB

            if trt_version == TRT8:
                if FP16:
                    config.set_flag(trt.BuilderFlag.FP16)
            else:
                builder.fp16_mode = FP16

            with open(model_file, "rb") as model:
                parser.parse(model.read())
            if verbose:
                logger.info(">" * 50)
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))

            # network.get_input(0).shape = [batch_size, 3, 800, 800]
            if chw_shape:
                network.get_input(0).shape = [batch_size, *chw_shape]

            if dynamic_input:
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    "inputs", (1, 3, 800, 800), (8, 3,
                                                 800, 800), (64, 3, 800, 800)
                )
                config.add_optimization_profile(profile)

            # builder engine
            engine = None
            if trt_version == TRT8:
                engine = builder.build_engine(network, config)
            else:
                engine = builder.build_cuda_engine(network)

            if engine is not None:
                logger.info("[INFO] Completed creating Engine!")
                with open(engine_file, "wb") as f:
                    f.write(engine.serialize())
            else:
                logger.error("Create engine failed!")
            return engine

    if os.path.exists(engine_file):
        # If a serialized engine exists, use it instead of building an engine.
        logger.info("[INFO] Reading engine from file {}".format(engine_file))
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return get_engine()


# int8 quant
def build_engine_onnx_v2(
    onnx_file_path="",
    engine_file_path="",
    fp16_mode=False,
    int8_mode=False,
    max_batch_size=1,
    calibration_stream=None,
    calibration_table_path="",
    save_engine=False,
    opt_params: dict = None,
):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine, opt_params=None):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, builder.create_builder_config() as trt_config:

            trt_version = int(trt.__version__[0])
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit(f"[Error]ONNX file {onnx_file_path} not found")
            logger.info(
                f"[INFO] Loading ONNX file from path {onnx_file_path}...")
            with open(onnx_file_path, "rb") as model:
                logger.info("[INFO] Beginning ONNX file parsing")
                parser.parse(model.read())
                assert (
                    network.num_layers > 0
                ), "[Error] Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible "
            logger.info("[INFO] Completed parsing of ONNX file")
            logger.info(
                f"[INFO] Building an engine from file {onnx_file_path}, this may take a while..."
            )

            # build trt engine

            # set optimize profile for dynamic inputs
            profile = builder.create_optimization_profile()
            if opt_params is not None:
                """
                opt_params = {
                    'input': [
                        [1, 3, 416, 502], # min shape
                        [1, 3, 416, 502], # opt shape
                        [1, 3, 416, 502], # max shape
                    ]
                }
                """
                logger.info('using opt_params: {}'.format(opt_params))
                for input_index, input_tensor_name in enumerate(opt_params.keys()):
                    min_shape = tuple(opt_params[input_tensor_name][0][:])
                    opt_shape = tuple(opt_params[input_tensor_name][1][:])
                    max_shape = tuple(opt_params[input_tensor_name][2][:])
                    profile.set_shape(
                        input_tensor_name, min_shape, opt_shape, max_shape
                    )
                    max_batch_size = max_shape[0]

                builder.max_batch_size = max_batch_size
            trt_config.add_optimization_profile(profile)

            if trt_version == TRT8:
                trt_config.max_workspace_size = 2 << 30  # 2GB
            else:
                builder.max_workspace_size = 2 << 30  # 2GB

            if trt_version == TRT8:
                if fp16_mode:
                    trt_config.set_flag(trt.BuilderFlag.FP16)
                    logger.info('enabled fp16 mode.')
            else:
                if fp16_mode:
                    builder.fp16_mode = fp16_mode
                    logger.info('enabled fp16 mode.')

            if int8_mode:
                if trt_version == TRT8:
                    trt_config.set_flag(trt.BuilderFlag.INT8)
                    logger.info('enabled int8 mode.')
                else:
                    builder.fp16_mode = fp16_mode

                assert (
                    calibration_stream
                ), "[Error] a calibration_stream should be provided for int8 mode"
                config.int8_calibrator = Calibrator(
                    calibration_stream, calibration_table_path
                )
                # builder.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
                logger.info("Int8 mode enabled")

            engine = None
            if trt_version == TRT8:
                engine = builder.build_engine(network, trt_config)
            else:
                engine = builder.build_cuda_engine(network)

            if engine is None:
                logger.info("Failed to create the engine")
                return None
            logger.info("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        logger.info(f"Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine, opt_params=opt_params)


def load_engine_from_local(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        logger.info(f"Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        logger.info("engine does not exist, please build it first.")
        exit(1)


def load_torchtrt_plugins():
    # ctypes.CDLL(osp.join(dir_path, 'libamirstan_plugin.so'))
    # suppose plugins lib installed into HOME
    lib_path = osp.join(
        osp.expanduser(
            "~"), "torchtrt_plugins/build/lib/libtorchtrt_plugins.so"
    )
    lib_path2 = '/usr/local/lib/libtorchtrt_plugins.so'
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
    elif os.path.exists(lib_path2):
        ctypes.CDLL(lib_path2)
    else:
        logger.warning(f"{lib_path} not found.")


def build_engine_onnx_v3(
    onnx_file_path="",
    fp16_mode=False,
    int8_mode=False,
    max_batch_size=1,
    calibration_stream=None,
    calibration_table_path="",
    save_engine=True,
):
    """
    this is deprecated.
    """
    engine_file_path = os.path.join(
        os.path.dirname(onnx_file_path),
        os.path.basename(onnx_file_path).replace(".onnx", ".engine"),
    )
    trt.init_libnvinfer_plugins(None, "")

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, builder.create_builder_config() as trt_config:

            trt_version = int(trt.__version__[0])
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit(f"[Error]ONNX file {onnx_file_path} not found")
            logger.info(f"Loading ONNX file from path {onnx_file_path}...")
            with open(onnx_file_path, "rb") as model:
                logger.info("Beginning ONNX file parsing")
                parser.parse(model.read())
                assert (
                    network.num_layers > 0
                ), "[Error] Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible "
            logger.info("[INFO] Completed parsing of ONNX file")
            logger.info(
                f"Building an engine from file {onnx_file_path}, this may take a while..."
            )

            # build trt engine
            builder.max_batch_size = max_batch_size

            if trt_version == TRT8:
                trt_config.max_workspace_size = 2 << 30  # 2GB
            else:
                builder.max_workspace_size = 2 << 30  # 2GB

            if trt_version == TRT8:
                if fp16_mode:
                    trt_config.set_flag(trt.BuilderFlag.FP16)
            else:
                builder.fp16_mode = fp16_mode

            if int8_mode:
                if trt_version == TRT8:
                    trt_config.set_flag(trt.BuilderFlag.INT8)
                else:
                    builder.fp16_mode = fp16_mode

                assert (
                    calibration_stream
                ), "[Error] a calibration_stream should be provided for int8 mode"
                config.int8_calibrator = Calibrator(
                    calibration_stream, calibration_table_path
                )
                # builder.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
                logger.info("Int8 mode enabled")

            engine = None
            if trt_version == TRT8:
                engine = builder.build_engine(network, trt_config)
            else:
                engine = builder.build_cuda_engine(network)

            if engine is None:
                logger.info("Failed to create the engine")
                return None
            logger.info("Completed creating the engine")
            if save_engine:
                logger.info(f"engine saved into: {engine_file_path}")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        logger.info(f"Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def check_engine(engine, input_shape=(608, 608), do_print=False):
    tensor_names_shape_dict = OrderedDict()
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        if dims[-1] == -1:
            assert (
                input_shape is not None
            ), "dynamic trt engine must specific input_shape"
            dims[-2], dims[-1] = input_shape
        size = trt.volume(dims) * engine.max_batch_size
        dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))

        if engine.binding_is_input(binding):
            tensor_names_shape_dict[binding] = {
                "shape": dims,
                "dtype": dtype,
                "is_input": True,
            }
            if do_print:
                print(f"[{binding}]: {dims}, {dtype.name}, [INPUT]")
        else:
            tensor_names_shape_dict[binding] = {
                "shape": dims,
                "dtype": dtype,
                "is_input": False,
            }
            if do_print:
                print(f"[{binding}]: {dims}, {dtype.name}, [OUTPUT]")
    return tensor_names_shape_dict
