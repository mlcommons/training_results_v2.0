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

import math
from mxnet.base import check_call, _LIB, c_array, _Null
from mxnet.gluon import nn, HybridBlock
import ctypes
from mpi4py import MPI
import mxnet as mx
import numpy as np

USE_MPI4PY = True

anti_gc = []


def handler_bytes():
    return 64


def _init_gbn_buffers(bn_group, local_rank, comm):
    assert bn_group >= 1, 'bn_group can\'t be smaller than 1'
    if bn_group == 1:
        return _Null

    sync_depth = int(math.log2(bn_group))  # required sync steps
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_gpus = local_comm.Get_size()
    xbuf_ptr = (ctypes.c_void_p * local_gpus)()
    handler = np.zeros(handler_bytes(), dtype=np.byte)
    check_call(_LIB.MXInitXBufSingle(local_rank, sync_depth, xbuf_ptr, handler.ctypes.data_as(ctypes.c_void_p)))

    handlers = np.zeros(handler_bytes()*local_gpus, dtype=np.byte)
    local_comm.Allgather([handler, handler_bytes(), MPI.BYTE], [handlers, handler_bytes(), MPI.BYTE])
    check_call(_LIB.MXOpenIpcHandles(local_rank, local_gpus, sync_depth, xbuf_ptr, handlers.ctypes.data_as(ctypes.c_void_p)))

    anti_gc.append(xbuf_ptr)
    return ctypes.addressof(xbuf_ptr)


class GroupInstanceNorm(HybridBlock):
    """
    Batch normalization layer (Ioffe and Szegedy, 2014) with GBN support.
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    bn_group : int, default 1
        Batch norm group size. if bn_group>1 the layer will sync mean and variance between
        all GPUs in the group. Currently only groups of 1, 2 and 4 are supported

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, in_channels=0, axis=-1, scale=True, center=True,
                 spatial_group_size=1, local_rank=0, comm=None, act_type='relu', **kwargs):
        super(GroupInstanceNorm, self).__init__(**kwargs)
        assert spatial_group_size in [1, 2, 4, 8]
        assert comm is not None
        if in_channels != 0:
            self.in_channels = in_channels

        # set parameters.
        self.c_max = 256

        self.xbuf_ptr = _init_gbn_buffers(bn_group=spatial_group_size, local_rank=local_rank, comm=comm)

        self.instance_norm = nn.InstanceNormV2(in_channels=in_channels,
                                              axis=axis,
                                              scale=scale,
                                              center=center,
                                              act_type=act_type,
                                              xbuf_ptr=self.xbuf_ptr,
                                              xbuf_group=spatial_group_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.instance_norm(x)
