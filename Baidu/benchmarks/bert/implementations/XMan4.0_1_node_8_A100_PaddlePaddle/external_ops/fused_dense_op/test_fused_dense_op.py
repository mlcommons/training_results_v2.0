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
from custom_setup_ops import custom_fused_dense
import numpy as np

iters = 1


def test_fused_dense_op(x, weight, bias, transx, transy, grad_out, atol):

    # x * weight^t + bias
    def run_paddle_mm_bias(x, weight, bias, transx, transy, grad_out):
        pp_x = paddle.to_tensor(x, stop_gradient=False)
        pp_weight = paddle.to_tensor(weight, stop_gradient=False)
        pp_bias = paddle.to_tensor(bias, stop_gradient=False)
        pp_out = paddle.matmul(pp_x, pp_weight, transx, transy)

        pp_bias_out = paddle.add(pp_out, pp_bias)

        pp_grad_out = paddle.to_tensor(grad_out)
        paddle.autograd.backward(
            [pp_bias_out], [pp_grad_out], retain_graph=True)

        return pp_bias_out, pp_x.grad, pp_weight.grad, pp_bias.grad

    def run_custom_fuse_dense(x, weight, bias, transx, transy, grad_out):
        x_tensor = paddle.to_tensor(x, stop_gradient=False)
        weight_tensor = paddle.to_tensor(weight, stop_gradient=False)
        bias_tensor = paddle.to_tensor(bias, stop_gradient=False)
        out_tensor = custom_fused_dense(x_tensor, weight_tensor, bias_tensor,
                                        transx, transy)

        grad_out_tensor = paddle.to_tensor(grad_out)
        paddle.autograd.backward(
            [out_tensor], [grad_out_tensor], retain_graph=True)

        return out_tensor, x_tensor.grad, weight_tensor.grad, bias_tensor.grad

    ''' 
    def run_numpy_mm_bias(x, weight, bias, transx, transy):
        out = np.matmul(x, weight.transpose(1, 0))
        out = out + bias
        return out
    '''

    def run_ref_backward(x, weight, bias, trans, transy, grad_out):
        x_tensor = paddle.to_tensor(x, stop_gradient=False)
        weight_tensor = paddle.to_tensor(weight, stop_gradient=False)
        grad_out_tensor = paddle.to_tensor(grad_out)
        # d_weight: x * grad_out^t (nt)
        # d_input: weight * grad_out (nn)
        if transy:
            ref_grad_weight = paddle.matmul(grad_out_tensor, x_tensor, True,
                                            False)
            ref_grad_input = paddle.matmul(grad_out_tensor, weight_tensor,
                                           False, False)
        else:
            ref_grad_weight = paddle.matmul(x_tensor, grad_out_tensor, True,
                                            False)
            ref_grad_input = paddle.matmul(grad_out_tensor, weight_tensor,
                                           False, True)
        ref_grad_bias = paddle.sum(grad_out_tensor, axis=0)
        return ref_grad_input, ref_grad_weight, ref_grad_bias

    '''
    ref_out = run_numpy_mm_bias(x, weight, bias, transx, transy)
    '''
    ref_pp_out, ref_pp_x_grad, ref_pp_weight_grad, ref_pp_bias_grad = run_paddle_mm_bias(
        x, weight, bias, transx, transy, grad_out)
    #print("ref_pp_x_grad shape: ", ref_pp_x_grad.shape)
    custom_out, x_grad, weight_grad, bias_grad = run_custom_fuse_dense(
        x, weight, bias, transx, transy, grad_out)
    #print("x_grad shape: ", x_grad.shape)

    new_ref_grad_input, new_ref_grad_weight, new_ref_grad_bias = run_ref_backward(
        x, weight, bias, transx, transy, grad_out)
    # check out
    '''
    np.testing.assert_allclose(
            ref_out, custom_out.numpy(), 
            rtol=1e-5, atol=atol)
    '''
    np.testing.assert_allclose(
        ref_pp_out.numpy(), custom_out.numpy(), rtol=1e-5, atol=atol)
    # check grad
    np.testing.assert_allclose(
        ref_pp_x_grad.numpy(), x_grad.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        new_ref_grad_weight.numpy(), weight_grad.numpy(), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(
        new_ref_grad_bias.numpy(), bias_grad.numpy(), rtol=1e-5, atol=atol)


def generate_input_data(m, dtype=np.float16):
    # index = np.random.randint(0, 5, (m))
    data = np.random.random((m)).astype(dtype)
    for i in range(m):
        #index[i] = 1
        #data[i] = 1.0/(np.exp2(index[i]))
        if i % 2 == 0:
            data[i] = 0.25
        elif i % 3 == 0:
            data[i] = 0.5
        else:
            data[i] = 0.0625
    return data


def generate_fixed_input(x_m, in_feature, out_feature, transy, dtype):
    x = generate_input_data(x_m * in_feature, dtype)
    x = x.reshape(x_m, in_feature)
    weight = generate_input_data(out_feature * in_feature, dtype)
    if transy:
        weight = weight.reshape(out_feature, in_feature)
    else:
        weight = weight.reshape(in_feature, out_feature)
    bias = generate_input_data(out_feature, dtype)
    grad_out = generate_input_data(x_m * out_feature, dtype)
    grad_out = grad_out.reshape(x_m, out_feature)

    return x, weight, bias, grad_out


def generate_ones_input(x_m, in_feature, out_feature, transy, dtype):
    x = np.ones(x_m * in_feature).astype(dtype)
    x = x.reshape(x_m, in_feature)
    weight = np.ones(out_feature * in_feature).astype(dtype)
    if transy:
        weight = weight.reshape(out_feature, in_feature)
    else:
        weight = weight.reshape(in_feature, out_feature)
    bias = np.ones(out_feature).astype(dtype)
    # bias = np.zeros(out_feature).astype(dtype)
    grad_out = np.ones(x_m * out_feature).astype(dtype)
    grad_out = grad_out.reshape(x_m, out_feature)

    return x, weight, bias, grad_out


def test_driver(i=0,
                x_m=56,
                in_feature=4,
                out_feature=8,
                transx=False,
                transy=True,
                dtype=np.float16,
                atol=1e-2):
    for i in range(iters):
        if i == 0:
            x, weight, bias, grad_out = generate_ones_input(
                x_m, in_feature, out_feature, transy, dtype)
        elif i == 1:
            x, weight, bias, grad_out = generate_fixed_input(
                x_m, in_feature, out_feature, transy, dtype)
        else:
            x = np.random.random((x_m, in_feature)).astype(dtype)
            if transy:
                weight = np.random.random(
                    (out_feature, in_feature)).astype(dtype)
            else:
                weight = np.random.random(
                    (in_feature, out_feature)).astype(dtype)
            bias = np.random.random((out_feature)).astype(dtype)
            grad_out = np.random.random((x_m, out_feature)).astype(dtype)

        test_fused_dense_op(x, weight, bias, transx, transy, grad_out, atol)


## Note: mlperf config: x_m from xx to 28672, in_feature is 1024, out_feature is 1024/4096. 
for i in range(3):
    print("Begin Test ", i)
    if i == 0 or i == 1:
        fp16_atol = 1e-5
        atol = 1e-5
    else:
        fp16_atol = 0.3
        atol = 1e-3

    #####################################################
    ## nt
    ## randome input: 0.2 is not right, should set to 0.3
    print("gemm_nt + bias test: ")
    test_driver(
        i,
        x_m=56,
        in_feature=1024,
        out_feature=1024,
        transx=False,
        transy=True,
        atol=fp16_atol)
    test_driver(
        i,
        x_m=56,
        in_feature=1024,
        out_feature=4096,
        transx=False,
        transy=True,
        atol=fp16_atol)
    test_driver(
        i,
        x_m=1000,
        in_feature=1024,
        out_feature=1024,
        transx=False,
        transy=True,
        atol=fp16_atol)
    ## for 0.0625 input, fp16 type's max error is 0.03125 
    test_driver(
        i,
        x_m=2000,
        in_feature=1024,
        out_feature=1024,
        transx=False,
        transy=True,
        dtype=np.float32,
        atol=atol)
    test_driver(
        i,
        x_m=28672,
        in_feature=1024,
        out_feature=1024,
        transx=False,
        transy=True,
        dtype=np.float32,
        atol=atol)
    test_driver(
        i,
        x_m=28672,
        in_feature=1024,
        out_feature=4096,
        transx=False,
        transy=True,
        dtype=np.float32,
        atol=atol)

    #####################################################
    ## nn 
    print("gemm_nn + bias test: ")
    test_driver(
        i,
        x_m=2,
        in_feature=1,
        out_feature=4,
        transx=False,
        transy=False,
        dtype=np.float32,
        atol=atol)
    test_driver(
        i,
        x_m=56,
        in_feature=1024,
        out_feature=1024,
        transx=False,
        transy=False,
        dtype=np.float32,
        atol=atol)
    test_driver(
        i,
        x_m=56,
        in_feature=1024,
        out_feature=4096,
        transx=False,
        transy=False,
        dtype=np.float32,
        atol=atol)
    test_driver(
        i,
        x_m=28672,
        in_feature=1024,
        out_feature=1024,
        transx=False,
        transy=False,
        dtype=np.float32,
        atol=atol)
    test_driver(
        i,
        x_m=28672,
        in_feature=1024,
        out_feature=4096,
        transx=False,
        transy=False,
        dtype=np.float32,
        atol=atol)
