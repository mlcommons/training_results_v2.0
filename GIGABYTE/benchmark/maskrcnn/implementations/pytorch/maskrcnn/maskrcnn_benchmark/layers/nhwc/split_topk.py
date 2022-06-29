# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch

# Faster (in some cases) implementation of topk.
# input is a 2D tensor of shape [N0, N1], dim =1.
# torch.topk will launch N0 CTA's, when N0 is small,
# i.e. much smaller than number of SMs on the GPU we
# get very poor performance since most of the SMs are
# idle. The WAR is to split N1 into num_chunks slices
# by reshaping input from [N0,N1] to [N0*num_chunks,N1/num_chunks],
# which increases number of CTAs launched from N0 to N0*num_chunks.
# Doing this means we have to do a second topk to merge the chunks,
# but overall this is much faster than original topk.
def split_topk(input, k, sorted=False, num_chunks=2):
    N0, N1 = list(input.shape)
    NN0, NN1 = N0*num_chunks, N1//num_chunks
    part_input = input.reshape([NN0,NN1])
    part_val, part_idx = torch.topk(part_input, k=k, dim=1)
    part_idx_offset = torch.tensor([NN1*i for i in range(num_chunks)]*N0, dtype=part_idx.dtype, pin_memory=True).to(device='cuda', non_blocking=True).reshape([NN0,1])
    part_idx.add_(part_idx_offset)
    part_val = part_val.reshape([N0,k*num_chunks])
    part_idx = part_idx.reshape([N0,k*num_chunks])
    top_val, idx2 = torch.topk(part_val, k=k, sorted=sorted, dim=1)
    top_idx = torch.gather(part_idx, 1, idx2)
    return top_val, top_idx
