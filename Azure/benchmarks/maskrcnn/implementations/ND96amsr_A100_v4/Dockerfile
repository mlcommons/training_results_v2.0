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
FROM ${FROM_IMAGE_NAME}

## Issue: if you need nsys, then you have to use this old version:
# 2021.3.1.57

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libibumad3 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/object_detection

# Uncomment the line below to control CUDA archs to support
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Control number of workers for ninja, so to avoid OOM when building
# The current build machine has 15GB memory, and peak memory usage per job is ~1.8GB
ENV MAX_JOBS 6

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy detectron code and build
COPY . .
RUN pip install -e maskrcnn
ENV OMP_NUM_THREADS=1
ENV OPENCV_FOR_THREADS_NUM=1
