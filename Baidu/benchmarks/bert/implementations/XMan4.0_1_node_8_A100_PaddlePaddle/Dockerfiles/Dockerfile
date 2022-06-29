# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#To get the latest APEX
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.04-py3

FROM ${FROM_IMAGE_NAME}

# Install dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bzip2 \
        cabextract \
        iputils-ping \
        pbzip2 \
        pv \
        lsof \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/bert
COPY requirements.txt .
ARG PYTHON=python3.8
RUN $PYTHON -m pip install --no-cache-dir -r requirements.txt

# Preprocessing
# WORKDIR /workspace
RUN cd /workspace && git clone https://github.com/attardi/wikiextractor.git
RUN cd /workspace/wikiextractor && git checkout e4abb4cbd019b0257824ee47c23dd163919b731b

# Install BERT
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
ENV PYTHONPATH "/workspace/bert"

# Install GCC 8.2
RUN apt-get update
RUN apt-get install -y libssl-dev ccache

RUN mkdir -p /workspace/temp_install_dir
WORKDIR /workspace/temp_install_dir 

COPY sanitizer_platform_limits_posix.h .
COPY sanitizer_platform_limits_posix.cc .
RUN wget --no-proxy -O gcc-8.2.0.tar.xz https://paddle-ci.gz.bcebos.com/gcc-8.2.0.tar.xz 
RUN tar -vxf gcc-8.2.0.tar.xz
WORKDIR gcc-8.2.0
RUN mv ../sanitizer_platform_limits_posix.h libsanitizer/sanitizer_common/sanitizer_platform_limits_posix.h
RUN mv ../sanitizer_platform_limits_posix.cc libsanitizer/sanitizer_common/sanitizer_platform_limits_posix.cc
RUN sed -i 's/ftp/http/g' ./contrib/download_prerequisites
RUN ./contrib/download_prerequisites
ARG GCC_NEW_DIR=/usr/local/gcc-8.2
ARG GCC_OLD_DIR=/usr/bin
RUN env LIBRARY_PATH="" ./configure --prefix=$GCC_NEW_DIR \
        --enable-threads=posix --disable-checking --disable-multilib \
        --enable-languages=c,c++
RUN env LIBRARY_PATH="" make -j `nproc`
RUN env LIBRARY_PATH="" make install -j `nproc`
COPY replace_gcc_symlink.sh . 
RUN bash -ex replace_gcc_symlink.sh "$GCC_NEW_DIR" "$GCC_OLD_DIR" 

# Install cmake 3.16.0
WORKDIR /workspace/temp_install_dir
RUN wget -O cmake-3.16.0.tar.gz https://cmake.org/files/v3.16/cmake-3.16.0.tar.gz
RUN tar -zvxf cmake-3.16.0.tar.gz
WORKDIR cmake-3.16.0
RUN ./bootstrap
RUN make -j `nproc`
RUN make install -j `nproc`
COPY replace_cmake_symlink.sh . 
RUN bash -ex replace_cmake_symlink.sh /usr/local/bin/cmake 

# Cleanup install dir 
WORKDIR /workspace/bert
RUN rm -rf /workspace/temp_install_dir

# Remove libsnappy-dev to avoid ld link error when compiling Paddle
RUN apt-get purge -y libsnappy-dev

# Remove protoc to avoid compilation error of Paddle
RUN mv /usr/bin/protoc /usr/bin/protoc.bak  
