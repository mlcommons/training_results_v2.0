# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from custom_setup_ops import custom_fused_dropout_residual_ln
import numpy as np

iters = 2


def test_fused_dropout_op(x, residual, ln_scale, ln_bias, epsilon, dropout_rate,
                          grad_out, atol):

    # layer_norm(dropout(x) + residual)
    def run_paddle_op(x, residual, ln_scale, ln_bias, epsilon, dropout_rate,
                      grad_out):
        pp_x = paddle.to_tensor(x, stop_gradient=False)
        pp_residual = paddle.to_tensor(residual, stop_gradient=False)
        pp_ln_scale = paddle.to_tensor(ln_scale, stop_gradient=False)
        pp_ln_bias = paddle.to_tensor(ln_bias, stop_gradient=False)
        pp_grad_out = paddle.to_tensor(grad_out)

        if dropout_rate > 0:
            pp_out = paddle.nn.functional.dropout(pp_x, dropout_rate)
            pp_add_out = paddle.add(pp_out, pp_residual)
        else:
            pp_add_out = paddle.add(pp_x, pp_residual)

        pp_out = paddle.nn.functional.layer_norm(
            pp_add_out, pp_add_out.shape[1:], pp_ln_scale, pp_ln_bias, epsilon)

        paddle.autograd.backward([pp_out], [pp_grad_out], retain_graph=True)

        return pp_out, pp_add_out, pp_x.grad, pp_residual.grad, pp_ln_scale.grad, pp_ln_bias.grad

    def run_custom_fuse_dropout_op(x, residual, ln_scale, ln_bias, epsilon,
                                   dropout_rate, grad_out):
        x_tensor = paddle.to_tensor(x, stop_gradient=False)
        residual_tensor = paddle.to_tensor(residual, stop_gradient=False)
        ln_scale_tensor = paddle.to_tensor(ln_scale, stop_gradient=False)
        ln_bias_tensor = paddle.to_tensor(ln_bias, stop_gradient=False)
        grad_out_tensor = paddle.to_tensor(grad_out)

        # Note: use the default config of dropout.
        is_test = False
        fix_seed = True
        is_upscale_in_train = True
        seed_val = 0
        out_tensor, dropout_mask, ln_mean, ln_var, dropout_residual_out = custom_fused_dropout_residual_ln(
            x_tensor, residual_tensor, ln_scale_tensor, ln_bias_tensor, epsilon,
            is_test, fix_seed, seed_val, is_upscale_in_train, dropout_rate)

        paddle.autograd.backward(
            [out_tensor], [grad_out_tensor], retain_graph=True)

        return out_tensor, dropout_residual_out, x_tensor.grad, residual_tensor.grad, ln_scale_tensor.grad, ln_bias_tensor.grad

    pp_out, pp_add_out, pp_x_grad, pp_residual_grad, pp_ln_scale_grad, pp_ln_bias_grad = run_paddle_op(
        x, residual, ln_scale, ln_bias, epsilon, dropout_rate, grad_out)

    out, add_out, x_grad, residual_grad, ln_scale_grad, ln_bias_grad = run_custom_fuse_dropout_op(
        x, residual, ln_scale, ln_bias, epsilon, dropout_rate, grad_out)

    # check out
    np.testing.assert_allclose(
        pp_out.numpy(), out.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        pp_add_out.numpy(), add_out.numpy(), rtol=1e-5, atol=atol)
    # check grad
    np.testing.assert_allclose(
        pp_x_grad.numpy(), x_grad.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        pp_residual_grad.numpy(), residual_grad.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        pp_ln_scale_grad.numpy(), ln_scale_grad.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        pp_ln_bias_grad.numpy(), ln_bias_grad.numpy(), rtol=1e-5, atol=atol)


def generate_input_data(m, dtype=np.float16):
    data = np.random.random((m)).astype(dtype)
    for i in range(m):
        if i % 2 == 0:
            data[i] = 0.25
        elif i % 3 == 0:
            data[i] = 0.5
        else:
            data[i] = 0.0625
    return data


def generate_fixed_input(x_m, in_feature, dtype):
    x = generate_input_data(x_m * in_feature, dtype)
    x = x.reshape(x_m, in_feature)
    residual = generate_input_data(x_m * in_feature, dtype)
    residual = residual.reshape(x_m, in_feature)

    ln_scale = generate_input_data(in_feature, dtype)
    ln_bias = generate_input_data(in_feature, dtype)

    grad_out = generate_input_data(x_m * in_feature, dtype)
    grad_out = grad_out.reshape(x_m, in_feature)

    return x, residual, ln_scale, ln_bias, grad_out


def generate_ones_input(x_m, in_feature, dtype):
    x = np.ones(x_m * in_feature).astype(dtype)
    x = x.reshape(x_m, in_feature)

    residual = np.ones(x_m * in_feature).astype(dtype)
    residual = residual.reshape(x_m, in_feature)

    ln_scale = np.ones(in_feature).astype(dtype)
    ln_bias = np.ones(in_feature).astype(dtype)

    grad_out = np.ones(x_m * in_feature).astype(dtype)
    grad_out = grad_out.reshape(x_m, in_feature)

    return x, residual, ln_scale, ln_bias, grad_out


def test_driver(i=0,
                x_m=56,
                in_feature=4,
                epsilon=1e-5,
                dropout_rate=0,
                dtype=np.float16,
                atol=1e-2):
    for j in range(iters):
        if i == 0:
            x, residual, ln_scale, ln_bias, grad_out = generate_ones_input(
                x_m, in_feature, dtype)
        elif i == 1:
            x, residual, ln_scale, ln_bias, grad_out = generate_fixed_input(
                x_m, in_feature, dtype)
        else:
            x = np.random.random((x_m, in_feature)).astype(dtype)
            residual = np.random.random((x_m, in_feature)).astype(dtype)

            ln_scale = np.random.random((in_feature)).astype(dtype)
            ln_bias = np.random.random((in_feature)).astype(dtype)

            grad_out = np.random.random((x_m, in_feature)).astype(dtype)

        test_fused_dropout_op(x, residual, ln_scale, ln_bias, epsilon,
                              dropout_rate, grad_out, atol)


## Note: mlperf config: x_m is xx to 28672, in_feature is 1024.
## Note: only dropout_rate=0 is tested.
for i in range(3):
    print("Begin Test ", i)
    atol = 1e-5
    if i == 0 or i == 1:
        fp16_atol = 1e-5
    else:
        fp16_atol = 1e-2
        atol = 1e-3

    # fp16
    test_driver(
        i,
        x_m=56,
        in_feature=1024,
        epsilon=1e-5,
        dropout_rate=0,
        dtype=np.float16,
        atol=fp16_atol)
    test_driver(
        i,
        x_m=28672,
        in_feature=1024,
        epsilon=1e-5,
        dropout_rate=0,
        dtype=np.float16,
        atol=fp16_atol)

    # fp32
    test_driver(
        i,
        x_m=28672,
        in_feature=1024,
        epsilon=1e-5,
        dropout_rate=0,
        dtype=np.float32,
        atol=atol)
