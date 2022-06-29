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


def generate_mask(attention_mask, unpad_fmha=False):
    if unpad_fmha:
        # 对[bs, max_seq_len]，每一行求和，代表获取每一行的实际seq_len（一维）。
        #seqlen = attention_mask.sum(dim=1).to(dtype=torch.int32).flatten()
        attention_mask_tmp = paddle.sum(attention_mask, axis=1)
        attention_mask_sum = paddle.cast(attention_mask_tmp, 'int32')
        seqlen = paddle.reshape(attention_mask_sum, [-1])
        print("seqlen is ", seqlen)

        # 把非零元的下标存储下来。
        #indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        attention_mask_1d = paddle.reshape(attention_mask, [-1])
        indices = paddle.nonzero(attention_mask_1d, as_tuple=False)
        indices = paddle.reshape(indices, [-1])

        # 当前batch的max cur_len_seq
        # maxseqlen = seqlen.max().item()
        maxseqlen_d = paddle.max(seqlen)
        # Note: use paddle.CUDAPinnedPlace() will cause the following errors:
        '''
        File "/usr/local/lib/python3.8/dist-packages/paddle/fluid/framework.py", line 2305, in __init__
        for frame in traceback.extract_stack():
        UnimplementedError: Unsupported place type `CUDAPinnedPlace` when casting paddle place to enum place. (at /limin29/Paddle/paddle/fluid/framework/custom_tensor_utils.h:135)
        [operator < custom_fmha > error]
        '''
        # maxseqlen = paddle.tensor.creation._memcpy(maxseqlen_d, paddle.CUDAPinnedPlace())
        maxseqlen = paddle.tensor.creation._memcpy(maxseqlen_d,
                                                   paddle.CPUPlace())
        print("maxseqlen", maxseqlen)

        prefix_sum = paddle.cumsum(seqlen, axis=0)
        zero_tensor = paddle.zeros([1], dtype='int32')
        # 返回数组前缀和。[0, a[0], a[0] + a[1], ...]
        cu_seqlens = paddle.concat(x=[zero_tensor, prefix_sum])
        # 返回cu_seqlens最后一个元素，代表当前batch的所有实际seq_len之和。
        # device tensor with shape [1]
        ntokens_d = cu_seqlens[-1]
        # host tensor with shape [1]
        #ntokens = paddle.tensor.creation._memcpy(ntokens_d, paddle.CUDAPinnedPlace()) 
        ntokens = paddle.tensor.creation._memcpy(ntokens_d, paddle.CPUPlace())
        print("ntokens = ", ntokens)
        return indices, attention_mask, seqlen, ntokens, cu_seqlens, seqlen, maxseqlen
    else:
        raise NotImplementedError()
