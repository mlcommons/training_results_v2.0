# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import torch
import paddle

import torch.nn.functional as F

bsz = 32
hidden_size = 256
x_type = np.float16

hidden_states = np.random.rand(bsz, hidden_size).astype(x_type) * 0.25
Wq = np.random.rand(hidden_size, hidden_size).astype(x_type) * 0.25
Wk = np.random.rand(hidden_size, hidden_size).astype(x_type) * 0.25
Wv = np.random.rand(hidden_size, hidden_size).astype(x_type) * 0.25
Bq = np.random.rand(hidden_size).astype(x_type) * 0.25
Bk = np.random.rand(hidden_size).astype(x_type) * 0.25
Bv = np.random.rand(hidden_size).astype(x_type) * 0.25

Wqkv = np.concatenate((Wq, Wk, Wv))
Bqkv = np.concatenate((Bq, Bk, Bv))


def run_paddle_linear():
    paddle.disable_static(place=paddle.CUDAPlace(0))
    paddle.set_default_dtype(x_type)
    '''
    out = paddle.matmul(
            paddle.cast(paddle.to_tensor(hidden_states), 'float32'), 
            paddle.cast(paddle.to_tensor(Wqkv), 'float32'), 
            transpose_x=False, transpose_y=True)
    out = out + paddle.cast(paddle.to_tensor(Bqkv), 'float32')
    '''
    out = paddle.matmul(
        paddle.to_tensor(hidden_states),
        paddle.to_tensor(Wqkv),
        transpose_x=False,
        transpose_y=True)
    out = out + paddle.to_tensor(Bqkv)
    return out


def run_pytorch_linear():
    out = F.linear(
        torch.from_numpy(hidden_states).cuda(),
        torch.from_numpy(Wqkv).cuda(), torch.from_numpy(Bqkv).cuda())
    #out = F.linear(torch.from_numpy(hidden_states).cuda(), torch.from_numpy(Wqkv).cuda(), None)
    #out = out + torch.from_numpy(Bqkv).cuda()
    return out


def run_numpy_linear():
    out = np.matmul(hidden_states, Wqkv.transpose(1, 0))
    out = out + Bqkv
    return out


paddle_out = run_paddle_linear()
pytorch_out = run_pytorch_linear()
np_out = run_numpy_linear()

print("compare with pytorch:")
np.testing.assert_allclose(
    pytorch_out.cpu().detach().numpy(),
    paddle_out.numpy(),
    rtol=1e-5,
    atol=1e-2)
print("Success!")
print("paddle compare with numpy:")
np.testing.assert_allclose(np_out, paddle_out.numpy(), rtol=1e-5, atol=1e-2)
print("Success!")
print("pytorch compare with numpy:")
np.testing.assert_allclose(
    np_out, pytorch_out.cpu().detach().numpy(), rtol=1e-5, atol=1e-2)
print("Success!")
