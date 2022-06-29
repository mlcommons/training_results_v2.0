# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest

import os
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.utils.cpp_extension import load

paddle.enable_static()

map_np_dtype_to_fluid_dtype = {
    'bool': "bool",
    'int8': "int8",
    'uint8': "uint8",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


def np_dtype_to_fluid_str(dtype: np.dtype) -> str:
    return map_np_dtype_to_fluid_dtype[dtype.name]


def load_custom_ops():
    # for successful compilation using `paddle.utils.cpp_extension.load`
    # we need a empty paddle custom op which defined in `custom_nop_op.cc`
    # the custom popart pattern is defined in `custom_popart_pattern.cc`
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir += "/.."
    custom_dir = cur_dir + "/custom_ops"
    custom_ops = load(
        name="custom_ops",
        sources=[
            f"{custom_dir}/custom_ops.cc", f"{custom_dir}/attention_mask.cpp",
            f"{custom_dir}/cast_from_fp8.cpp", f"{custom_dir}/cast_to_fp8.cpp",
            f"{custom_dir}/disable_attn_dropout_bwd_pattern.cpp",
            f"{custom_dir}/dropout_with_switchx.cpp",
            f"{custom_dir}/embedding_gather.cpp",
            f"{custom_dir}/seed_modify.cpp",
            f"{custom_dir}/lamb_serialised_weight_pattern.cpp",
            f"{custom_dir}/sparse_accumulate_pattern.cpp",
            f"{custom_dir}/tied_gather_pattern.cpp",
            f"{custom_dir}/detach_shape_inference.cpp",
            f"{custom_dir}/workarounds/accumulate_priority_pattern.cpp",
            f"{custom_dir}/workarounds/improved_sum.cpp",
            f"{custom_dir}/workarounds/prevent_const_expr_folding_op.cpp"
        ],
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
        extra_ldflags=["-lpopfloat"])
    return custom_ops


class TestBase(unittest.TestCase):
    def setUp(self):
        self.set_feed()
        self.set_feed_attr()
        self.set_op()

    def set_op(self):
        # setup custom op
        self.saved_file = 'fp8_checkpointoutput.onnx'

    def set_feed(self):
        self.feed = {
            "x":
            np.array([[[1], [3]], [[2], [4]], [[4], [127]]]).astype('int64'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def test_base(self):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype='int64')

                out = paddle.fluid.layers.embedding(
                    x, size=[128, 16], dtype="float32")

                cast_to_fp8 = custom_ops.custom_CastToFp8(out, "4", "3", "7")
                checkpoint = custom_ops.checkpointoutput(cast_to_fp8)
                cast_from_fp8 = custom_ops.custom_CastFromFp8(
                    checkpoint, "FLOAT", "4", "3", "7")

                loss = paddle.mean(cast_from_fp8)
                adam = paddle.optimizer.Adam(learning_rate=1e-2)
                adam.minimize(loss)
                fetch_list = [loss.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            paddle.static.save(main_prog, "fp8")

            feed_list = [x.name]
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(
                num_ipus=1, micro_batch_size=1, is_training=True)
            ipu_strategy.add_custom_op("custom_CastFromFp8", "CastFromFp8",
                                       "ai.graphcore")
            ipu_strategy.add_custom_op("custom_CastToFp8", "CastToFp8",
                                       "ai.graphcore")
            ipu_compiler = paddle.static.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy)

            program = ipu_compiler.compile(feed_list, fetch_list)
            res = exe.run(program, self.feed, fetch_list)
            print(res)


if __name__ == "__main__":
    custom_ops = load_custom_ops()
    unittest.main()
