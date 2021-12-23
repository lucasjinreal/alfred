
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


class TensorRTInferencer:
    def __init__(self, engine_f, timing=False) -> None:
        self.engine_f = engine_f
        self.timing = timing
        self._init_engine()

    def _init_engine(self):
        self.engine = load_engine_from_local(self.engine_f)
        self.context = self.engine.create_execution_context()

        sps = check_engine(self.engine)
        self.input_shapes = [sps[i]["shape"] for i in sps.keys() if sps[i]["is_input"]]
        self.output_shapes = [
            sps[i]["shape"] for i in sps.keys() if not sps[i]["is_input"]
        ]
        self.ori_input_shape = self.context.get_binding_shape(0)
        self.is_dynamic_batch = self.ori_input_shape[0] == -1
        self.is_dynamic_shape = self.ori_input_shape[-1] == -1
        if self.is_dynamic_batch:
            logger.info(
                f"engine is dynamic on batch. engine max_batchsize: {self.engine.max_batch_size}"
            )
        if self.is_dynamic_shape:
            logger.info("engine is dynamic on shape.")

        if self.is_dynamic_shape:
            (
                self.inputs,
                self.outputs,
                self.bindings,
                self.stream,
            ) = allocate_buffers_v2_dynamic(self.engine)
        else:
            (
                self.inputs,
                self.outputs,
                self.bindings,
                self.stream,
            ) = allocate_buffers_v2(self.engine)

        self.context.set_optimization_profile_async(0, self.stream.handle)
        print("TRT engine loaded.")

    def infer(self, imgs):
        assert isinstance(imgs, np.ndarray), "imgs must be numpy array"

        if self.is_dynamic_batch:
            bs = imgs.shape[0]
            self.inputs[0].host = imgs.ravel()
            # make context aware what's batchsize current
            self.ori_input_shape[0] = bs
            self.context.set_binding_shape(0, (self.ori_input_shape))

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
        if self.timing:
            t1 = time.perf_counter()
            print(f"engine cost: {t1 - t0}, fps: {1/(t1-t0)}")
        outs_reshaped = []
        for i, o in enumerate(outs):
            o_s = self.output_shapes[i]
            if self.is_dynamic_batch:
                o_s[0] = self.engine.max_batch_size
            o = np.reshape(o, o_s)
            outs_reshaped.append(o[:bs, ...])
        return outs_reshaped

    def __del__(self):
        del self.engine
        del self.context
