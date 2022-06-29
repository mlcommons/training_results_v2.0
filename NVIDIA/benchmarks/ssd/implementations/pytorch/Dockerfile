# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.04-py3

################################################################################
# Builder image
################################################################################
FROM ${FROM_IMAGE_NAME} AS builder
WORKDIR /workspace/csrc
COPY csrc .
# Build CUDA proposal matcher (written as a DALI extension)
RUN cd dali_box_iou \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make
# Build CUDA box iou (written as a DALI extension)
RUN cd dali_proposal_matcher \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make
################################################################################

FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/ssd

# Copy code
COPY . .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Update container's pycocotools to the nvidia optimized version
# TODO(ahmadki): This should be merged with https://github.com/NVIDIA/cocoapi
RUN pip uninstall -y pycocotools
ENV COCOAPI_VERSION=2.0+nv0.6.1
RUN export COCOAPI_TAG=$(echo ${COCOAPI_VERSION} | sed 's/^.*+n//') \
 && pip install --no-cache-dir pybind11                             \
 && pip install --no-cache-dir git+https://github.com/ahmadki/cocoapi.git@ahmadki/mlperf_v2.0#subdirectory=PythonAPI

COPY --from=builder /workspace/csrc/dali_box_iou/build/lib_box_iou.so /usr/local/lib/lib_box_iou.so
COPY --from=builder /workspace/csrc/dali_proposal_matcher/build/lib_proposal_matcher.so /usr/local/lib/lib_proposal_matcher.so

# Set env variables
ENV CUDNN_V8_API_ENABLED=1

