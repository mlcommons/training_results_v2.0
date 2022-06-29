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
from custom_setup_ops import custom_fmha
import numpy as np

total = 2
batch_size = 56
num_heads = 16
head_size = 64

is_training = True
max_seq_len = 512
dropout_rate = 0.1

cu_seqlen = np.arange(batch_size + 1)
cu_seqlen = np.cumsum(cu_seqlen)
total = cu_seqlen[-1]
#print("cu_seqlen", cu_seqlen)
#print("cu_seqlen[-1]", cu_seqlen[-1])
cu_seqlen = paddle.to_tensor(cu_seqlen)
cu_seqlen = paddle.cast(cu_seqlen, 'int32')

qkv = np.random.random((total, 3, num_heads, head_size)).astype(np.float16)
#print("qkv:", qkv)
qkv = paddle.to_tensor(qkv, stop_gradient=False)

max_seq_len_host = [max_seq_len]
max_seq_len_host = paddle.to_tensor(
    max_seq_len_host, dtype='int32', place=paddle.CPUPlace())
ctx_out, s_out = custom_fmha(qkv, cu_seqlen, max_seq_len_host, is_training,
                             dropout_rate, False)
print("print ctx_out and s_out: ")
print(ctx_out)
print(s_out)

# backward.
print("print qkv.grad: ")
grad_ctx_dout = np.random.random(
    (total, num_heads, head_size)).astype(np.float16)
grad_ctx_dout = paddle.to_tensor(grad_ctx_dout)
paddle.autograd.backward([ctx_out], [grad_ctx_dout], retain_graph=True)
print(qkv.grad)
