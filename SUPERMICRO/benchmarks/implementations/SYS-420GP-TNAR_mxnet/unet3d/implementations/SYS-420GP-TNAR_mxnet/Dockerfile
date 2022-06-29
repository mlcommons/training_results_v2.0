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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:22.04-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

#RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0
ENV MXNET_ENABLE_CUDA_GRAPHS=1

# New dependency engine
ENV MXNET_ASYNC_GPU_ENGINE=1
# Fix for horovod with the new dependency engine
ENV HOROVOD_ENABLE_ASYNC_COMPLETION=1

# Fix HCOLL errors
ENV OMPI_MCA_coll_hcoll_enable=0

ENV MXNET_EXEC_ENABLE_ADDTO=1
ENV MXNET_INSTANCENORM_USE_DUAL_STREAMS=1
ENV MXNET_OPTIMIZER_AGGREGATION_SIZE=60
ENV MXNET_GPU_WORKER_NTHREADS=1
ENV MXNET_GPU_COPY_NTHREADS=1
ENV MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
ENV MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
ENV MXNET_CUDNN_NHWC_BN_HEURISTIC_FWD=1
ENV MXNET_CUDNN_NHWC_BN_HEURISTIC_BWD=1
ENV MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=1
ENV MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=1

# DALI jitter
ENV MALLOC_ARENA_MAX=1
ENV MALLOC_MMAP_MAX_=0
