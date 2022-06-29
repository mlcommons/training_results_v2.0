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
    custom_dir = f"{os.path.dirname(cur_dir)}/custom_ops"
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
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'])
    return custom_ops


class TestBase(unittest.TestCase):
    def setUp(self):
        self.set_feed()
        self.set_feed_attr()
        self.set_attrs()
        self.set_op()

    def set_op(self):
        # setup custom op
        self.op = custom_ops.custom_Detach
        self.saved_file = 'custom_Detach.onnx'

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(
                0, 1, size=[1, 2, 4, 4]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def set_attrs(self):
        self.attrs = {'pass_through_creation': 1}

    def test_base(self):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                feeds = []
                for idx in range(len(self.feed_list)):
                    x = paddle.static.data(
                        name=self.feed_list[idx],
                        shape=self.feed_shape[idx],
                        dtype=self.feed_dtype[idx])
                    feeds.append(x)

                # use custom op
                if self.op in [
                        custom_ops.custom_Detach,
                        custom_ops.checkpointoutput,
                        custom_ops.custom_DropoutWithTrainingSwitch,
                        custom_ops.custom_EmbeddingGather,
                ]:
                    feeds[0] = paddle.fluid.layers.conv2d(
                        feeds[0], 3, 3, padding=1)
                    if self.op == custom_ops.checkpointoutput:
                        out = self.op(*feeds)
                    else:
                        out = self.op(*feeds, **self.attrs)
                    out = paddle.fluid.layers.elementwise_add(out, out)
                elif self.op in [custom_ops.custom_AttentionMask]:
                    feeds[1] = paddle.fluid.layers.conv2d(
                        feeds[1], 4, 3, padding=1)
                    out = self.op(*feeds, **self.attrs)
                    out = paddle.fluid.layers.elementwise_add(out, out)

                loss = paddle.mean(out)
                fetch_list = [out.name]

                # apply optimizer
                adam = paddle.optimizer.Adam(learning_rate=1e-2)
                adam.minimize(loss)

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            feed_list = self.feed_list
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=True)
            if self.op == custom_ops.custom_DropoutWithTrainingSwitch:
                ipu_strategy.disable_pattern("OpToIdentity")
            ipu_strategy.add_custom_op("custom_AttentionMask", "AttentionMask",
                                       "ai.graphcore")
            ipu_strategy.add_custom_op("custom_Detach", "Detach", "custom.ops")
            ipu_strategy.add_custom_op("custom_DropoutWithTrainingSwitch",
                                       "DropoutWithTrainingSwitch",
                                       "ai.graphcore")
            ipu_strategy.add_custom_op("custom_EmbeddingGather",
                                       "EmbeddingGather", "ai.graphcore")
            ipu_compiler = paddle.static.IpuCompiledProgram(
                main_prog, ipu_strategy=ipu_strategy)
            program = ipu_compiler.compile(feed_list, fetch_list)

            for _ in range(10):
                loss = exe.run(program, feed=self.feed, fetch_list=fetch_list)
                print('loss: ', np.asarray(loss).shape)


class TestCase1(TestBase):
    def set_op(self):
        self.op = custom_ops.checkpointoutput
        self.saved_file = 'checkpointoutput.onnx'

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(
                0, 100, size=[1, 4, 4, 4]).astype('float32'),
        }


class TestCase2(TestBase):
    def set_op(self):
        self.op = custom_ops.custom_AttentionMask
        self.saved_file = 'custom_AttentionMask.onnx'

    def set_attrs(self):
        self.attrs = {'datatype': 'FLOAT'}

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(
                0, 100, size=[2, 512]).astype('int32'),
            "y": np.random.uniform(
                0, 1, size=[1, 4, 512, 512]).astype('float32'),
        }


class TestCase3(TestBase):
    def set_op(self):
        self.op = custom_ops.custom_DropoutWithTrainingSwitch
        self.saved_file = 'custom_DropoutWithTrainingSwitch.onnx'

    def set_attrs(self):
        self.attrs = {'ratio': 0.1}

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(
                0, 1, size=[1, 3, 5, 5]).astype('float32'),
            "y": np.ones([1]).astype('int32'),
        }


class TestCase4(TestBase):
    def set_op(self):
        self.op = custom_ops.custom_EmbeddingGather
        self.saved_file = 'custom_EmbeddingGather.onnx'

    def set_attrs(self):
        self.attrs = {'axis': 0}

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(
                0, 1, size=[1, 3, 4, 4]).astype('float32'),
            "y": np.random.uniform(
                0, 100, size=[4]).astype('int32'),
        }


if __name__ == "__main__":
    custom_ops = load_custom_ops()
    unittest.main()
