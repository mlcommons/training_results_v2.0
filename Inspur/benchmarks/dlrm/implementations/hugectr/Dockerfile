# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

ARG SM="80"
ARG ENABLE_MULTINODES=ON

ARG RAPIDS_VERSION=21.06
ARG HWLOC_VERSION=2.4.1
ARG RELEASE=true

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        clang-format \
	libboost-serialization-dev \
        libtbb-dev \
        libaio-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/bin:$PATH

# mpi4py
RUN pip3 install --no-cache-dir mpi4py

# CUDA-Aware hwloc
RUN cd /opt/hpcx/ompi/include/openmpi/opal/mca/hwloc/hwloc201 && rm -rfv hwloc201.h hwloc/include/hwloc.h
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.open-mpi.org/release/hwloc/v2.4/hwloc-${HWLOC_VERSION}.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/hwloc-${HWLOC_VERSION}.tar.gz -C /var/tmp && \
    cd /var/tmp/hwloc-${HWLOC_VERSION} && \
    ./configure CPPFLAGS="-I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/" LDFLAGS="-L/usr/local/cuda/lib64" --enable-cuda && \
    make -j$(nproc) && make install && \
    rm -rf /var/tmp/hwloc-${HWLOC_VERSION} /var/tmp/hwloc-${HWLOC_VERSION}.tar.gz

# Install Hiredis from source
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 https://github.com/redis/hiredis.git && cd - && \
    cd /var/tmp/hiredis && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && make install && \
    rm -rf /var/tmp/hiredis

# Install redis++ from source
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 -b 1.3.2 https://github.com/sewenew/redis-plus-plus.git && cd - && \
    cd /var/tmp/redis-plus-plus && \
    mkdir build && cd build && \
    cmake -DREDIS_PLUS_PLUS_CXX_STANDARD=17 .. && \
    make -j$(nproc) && make install && \
    rm -rf /var/tmp/redis-plus-plus

# Install RocksDB from source
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 -b v6.25.3 https://github.com/facebook/rocksdb.git && cd - && \
    cd /var/tmp/rocksdb && \
    PORTABLE=1 make -j$(nproc) shared_lib && \
    make install-shared && \
    rm -rf /var/tmp/rocksdb
ENV CPATH=/usr/local/include:$CPATH

# Install LibRdKafka from source
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends libssl-dev libsasl2-dev liblz4-dev libzstd-dev && \
    mkdir -p /var/tmp && cd /var/tmp && git clone --depth 1 -b v1.8.2 https://github.com/edenhill/librdkafka.git && cd - && \
    cd /var/tmp/librdkafka && \
    ./configure && make -j$(nproc) && make install && \
    rm -rf /var/tmp/librdkafka

# Env variables for NCCL
ENV NCCL_LAUNCH_MODE=PARALLEL \
    NCCL_COLLNET_ENABLE=0

# ENV variables for Sharp
ENV SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 \
    SHARP_COLL_LOCK_ON_COMM_INIT=1 \
    SHARP_COLL_LOG_LEVEL=3 \
    HCOLL_ENABLE_MCAST=0

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.36.0 /usr/lib/x86_64-linux-gnu/libibverbs.so

WORKDIR /workspace/dlrm
COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

# HugeCTR
RUN if [ "$RELEASE" = "true" ]; \
    then \
	cd /workspace/dlrm/ && \
    	git clone --recurse-submodules --branch=v3.7_preview https://github.com/NVIDIA-Merlin/HugeCTR.git hugectr && \
    	cd hugectr && \
    	mkdir build && cd build && \
    	cmake -DCMAKE_BUILD_TYPE=Release -DSM=$SM \
    	    -DENABLE_MULTINODES=$ENABLE_MULTINODES -DSHARP_A2A=OFF -DDISABLE_CUDF=ON .. && \
    	make -j$(nproc) && make install && \
    	chmod +x /usr/local/hugectr/bin/* && \
    	chmod +x /usr/local/hugectr/lib/* && \
    	rm -rf /workspace/dlrm/hugectr && \
	rm /usr/lib/x86_64-linux-gnu/libibverbs.so; \
    else \
      echo "Build container for development successfully"; \
    fi

ENV PATH=/usr/local/hugectr/bin:$PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
