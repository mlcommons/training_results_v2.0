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

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from mxnet.base import check_call, _LIB


class _SpatialParallelHelper(object):
    _init = False
    nccl_id = None
    num_gpus = None
    rank = None

    @staticmethod
    def init(num_gpus):
        """Communicate the NCCL unique id"""
        cls = _SpatialParallelHelper
        if not cls._init:
            cls._init = True
            import ctypes
            try:
                from mpi4py import MPI
            except:
                raise ImportError("Spatial parallel modules require mpi4py package.")
            import numpy as np
            global_comm = MPI.COMM_WORLD
            rank = global_comm.rank
            color = rank / num_gpus
            comm = global_comm.Split(color, rank)
            nccl_id_size = ctypes.c_int()
            check_call(_LIB.MXNCCLGetUniqueIdSize(ctypes.byref(nccl_id_size)))
            nccl_id_size = nccl_id_size.value
            cls.nccl_id = np.zeros(nccl_id_size, np.byte)
            if comm.rank == 0:
                check_call(_LIB.MXNCCLGetUniqueId(
                    cls.nccl_id.ctypes.data_as(ctypes.c_void_p)))
            comm.Bcast([cls.nccl_id, nccl_id_size, MPI.BYTE], root=0)
            cls.num_gpus = num_gpus
            cls.rank = rank % num_gpus
        assert num_gpus == cls.num_gpus, ("All of the spatial parallel "
                                          "operations need to span the same number of GPUs")
