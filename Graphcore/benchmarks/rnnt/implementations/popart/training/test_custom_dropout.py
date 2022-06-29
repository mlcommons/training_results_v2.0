import argparse
import numpy as np
import popart
import custom_op_utils
import logging_util
from test_utils import helper_run
from transducer_train import _get_popart_type as _get_popart_type

# set up logging
logger = logging_util.get_basic_logger('TEST_CUSTOM_OP_JOINT_NET')

def build_custom_dropout(builder, x, dropout_rate):
    logger.debug("Building custom dropout")
    y = builder.customOp(
            opName = "CustomDropout",
            opVersion = 1,
            domain = "com.acme",
            inputs = [x],
            attributes = {
                "ratio": dropout_rate
            },
            numOutputs = 1)[0]
    return y


def build_dropout_model(builder, input_values, dropout_rate, splits, custom):
    x_cpu = input_values
    x = builder.addInputTensor(popart.TensorInfo(_get_popart_type(x_cpu.dtype.type), x_cpu.shape), "x")
    
    assert(splits > 0)
    if splits > 1:
        T = x_cpu.shape[1]
        split_array = [T // splits] * splits
        t_splits = builder.aiOnnx.split([x], num_outputs=len(split_array),
                                             axis=1,
                                             split=split_array)
        y_splits = []
        for t_part in t_splits:
            with builder.virtualGraph(0):
                if custom:
                    y_part = build_custom_dropout(builder, t_part, dropout_rate)
                else:
                    y_part = builder.aiOnnx.dropout([t_part], 1, dropout_rate)[0]
                y_splits.append(y_part)
            y = builder.aiOnnx.concat(y_splits, axis=1)
    else:
        if custom:
            y = build_custom_dropout(builder, x, dropout_rate)
        else:
            y = builder.aiOnnx.dropout([x], 1, dropout_rate)[0]

    builder.addOutputTensor(y)

    dx = popart.reservedGradientPrefix() + x

    loss = builder.aiOnnx.reducesum([y])
    optimizer = popart.SGD({"defaultLearningRate": (0.1, False)})

    return (x, y, dx, loss, optimizer)


def dropout(T, U, A, B, dropout_rate, t_splits, custom, outline):
    logger.debug("\ncustom: {}, outline: {}".format(custom, outline))

    custom_op_utils.load_custom_lib("dropout")
    builder = popart.Builder()

    x_cpu = np.ones([B, T, U, A]).astype(np.float32)

    (x, y, dx, loss, optimizer) = build_dropout_model(builder, x_cpu, dropout_rate, t_splits, custom)

    (y_cpu, dx_cpu) = helper_run(builder, (x), (x_cpu), (y, dx), 1, True, loss, optimizer, outline, 0)

    y_cpu_d = y_cpu.reshape([B * t_splits, (T // t_splits) * U * A])
    dx_cpu_d = dx_cpu.reshape([B * t_splits, (T // t_splits) * U * A])
    
    logger.debug("y:\n{}".format(y_cpu_d))
    logger.debug("dx:\n{}".format(dx_cpu_d))

    return y_cpu, dx_cpu

"""
    Test custom dropout op in isolation.
    The test verifies that forward and backward pass masks are thensame.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=8,
                        help='T dimension')
    parser.add_argument('--U', type=int, default=8,
                        help='U dimension')
    parser.add_argument('--A', type=int, default=1,
                        help='A dimension')
    parser.add_argument('--B', type=int, default=1,
                        help='B dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--t-splits', type=int, default=1,
                        help='Splits along T dimension')
    parser.add_argument('--base', action="store_true", default=False,
                        help='Check base dropout op')
    parser.add_argument('--outline', action="store_true", default=False,
                        help='Whether to do outlining')
    conf = parser.parse_args()
    conf.custom = not conf.base

    y_cpu, dx_cpu = dropout(conf.T, conf.U, conf.A, conf.B, conf.dropout_rate, conf.t_splits, conf.custom, conf.outline)
    assert(np.array_equal(y_cpu, dx_cpu))

    logger.info("All test passed")
