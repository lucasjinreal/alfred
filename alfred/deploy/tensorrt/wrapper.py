
from alfred.deploy.tensorrt.common import (
    do_inference_v2,
    build_engine_onnx_v2,
    allocate_buffers_v2_dynamic,
    allocate_buffers_v2,
    load_engine_from_local,
    load_torchtrt_plugins,
    check_engine,
)
import numpy as np
from alfred.utils.log import logger
import time
import pycuda.driver as cuda


class TensorRTInferencer:
    def __init__(self, engine_f, device_id=0, cuda_ctx=None, timing=False) -> None:
        self.engine_f = engine_f
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()
        self.timing = timing
        self._init_engine()

    def _init_engine(self):
        self.engine = load_engine_from_local(self.engine_f)
        sps = check_engine(self.engine)
        self.input_shapes = [sps[i]["shape"]
                             for i in sps.keys() if sps[i]["is_input"]]
        self.output_shapes = [
            sps[i]["shape"] for i in sps.keys() if not sps[i]["is_input"]
        ]
        self.ori_input_shape = self.engine.get_binding_shape(0)
        self.is_dynamic_batch = self.ori_input_shape[0] == -1
        self.is_dynamic_shape = self.ori_input_shape[-1] == -1
        if self.is_dynamic_batch:
            logger.info(
                f"engine is dynamic on batch. engine max_batchsize: {self.engine.max_batch_size}"
            )
        if self.is_dynamic_shape:
            logger.info("engine is dynamic on shape.")

        try:
            self.context = self.engine.create_execution_context()
            if self.is_dynamic_shape:
                (
                    self.inputs,
                    self.outputs,
                    self.bindings,
                    self.stream,
                ) = allocate_buffers_v2_dynamic(self.engine)
                self.context.set_optimization_profile_async(
                    0, self.stream.handle)
            else:
                (
                    self.inputs,
                    self.outputs,
                    self.bindings,
                    self.stream,
                ) = allocate_buffers_v2(self.engine)
            print("TRT engine loaded.")
            if self.cuda_ctx:
                self.cuda_ctx.pop()
        except Exception as e:
            self.cuda_ctx.pop()
            del self.cuda_ctx
            raise RuntimeError("Fail to allocate CUDA resources") from e

    def infer(self, imgs):
        assert isinstance(imgs, np.ndarray), "imgs must be numpy array"
        if self.cuda_ctx:
            self.cuda_ctx.push()

        if self.is_dynamic_batch:
            bs = imgs.shape[0]
            self.inputs[0].host = imgs.ravel()
            # make context aware what's batchsize current
            self.ori_input_shape[0] = bs
            self.context.set_binding_shape(0, (self.ori_input_shape))
        else:
            self.inputs[0].host = imgs.ravel()

        if self.timing:
            t0 = time.perf_counter()
        outs = do_inference_v2(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            input_tensor=imgs,
        )
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        if self.timing:
            t1 = time.perf_counter()
            print(f"engine cost: {t1 - t0}, fps: {1/(t1-t0)}")
        outs_reshaped = []
        for i, o in enumerate(outs):
            o_s = self.output_shapes[i]
            if self.is_dynamic_batch:
                o_s[0] = self.engine.max_batch_size
            o = np.reshape(o, o_s)
            if self.is_dynamic_batch:
                outs_reshaped.append(o[:bs, ...])
            else:
                outs_reshaped.append(o)
        return outs_reshaped

    def __del__(self):
        """Free CUDA memories"""
        del self.outputs
        del self.inputs
        del self.stream
