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
from bert_padding import generate_mask

# test for generate_mask

# [3, 4]
attention_mask = [[0, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1]]
attention_mask_tensor = paddle.to_tensor(attention_mask)
indices, attention_mask, seqlen, ntokens, cu_seqlens, seqlen, maxseqlen = generate_mask(
    attention_mask_tensor, True)

print("indices", indices)
print("attention_mask", attention_mask)
print("seqlen", seqlen)
print("ntokens", ntokens)
print("cu_seqlens", cu_seqlens)
print("maxseqlen", maxseqlen)

# test for unpad(gather) and pad(scatter) 

embed_dim = 3
input = [[0, 0, 0], [1, 1, 1], [0, 0, 0], [3, 3, 3], [4, 4, 4]]
index = [1, 3, 4]
input = paddle.to_tensor(input)
index = paddle.to_tensor(index)
out = paddle.gather(input, index)

print(input)
print(index)
print("gather out", out)

batch_size = 1
max_seq_len = 5

scatter_shape = [batch_size * max_seq_len, embed_dim]
zero_tensor = paddle.zeros(scatter_shape, dtype='int64')
scatter_out = paddle.scatter(zero_tensor, index, out)
print("scatter_out: ", scatter_out)

# error: 
#out_1 = paddle.scatter_nd(index, out, scatter_shape)
#print("scatter_nd out: ", out_1)
