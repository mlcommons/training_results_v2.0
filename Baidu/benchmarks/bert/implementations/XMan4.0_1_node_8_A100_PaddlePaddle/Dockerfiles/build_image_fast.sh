# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

set -e

SRC_IMAGE="nvcr.io/nvidia/pytorch:22.04-py3"
DST_IMAGE="nvcr.io/nvidia/pytorch:22.04-py3-paddle-fast-test"

WHEEL_URL="https://paddle-wheel.bj.bcebos.com/mlperf-2.0"
PD_WHEEL_NAME="paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"
OP_TAR_NAME="custom_setup_ops.tar.gz"
PYBIND_FUNCTION_SO_NAME="functions.cpython-38-x86_64-linux-gnu.so"
PYTHON="python3.8"

###################

TMP_DOCKERFILE=Dockerfile.tmp
APEX_CLONE_DIR=/workspace/apex_dir
APEX_DIR=$APEX_CLONE_DIR/apex/build_scripts
PY_PACKAGE_DIR=/opt/conda/lib/python3.8/site-packages
OP_INSTALL_DIR=$PY_PACKAGE_DIR/custom_setup_ops

if [[ $SRC_IMAGE == $DST_IMAGE ]]; then
  echo "Error: SRC_IMAGE and DST_IMAGE cannot be the same!!!" 
  exit 1
fi

OLD_DIR=`pwd`
NEW_DIR=$(dirname `readlink -f "$0"`)
cd $NEW_DIR

cat <<EOF >$TMP_DOCKERFILE
FROM $SRC_IMAGE
RUN mkdir -p $APEX_CLONE_DIR \
        && cd $APEX_CLONE_DIR \
        && git clone -b new_fmhalib https://github.com/sneaxiy/apex \
        && cd $APEX_DIR \
        && bash build.sh 
RUN curl -O $WHEEL_URL/$PD_WHEEL_NAME \
        && $PYTHON -m pip install -U --force-reinstall $PD_WHEEL_NAME \
        && rm -rf $PD_WHEEL_NAME
RUN mkdir -p $OP_INSTALL_DIR \
        && cd $OP_INSTALL_DIR \
        && curl -O $WHEEL_URL/$OP_TAR_NAME \
        && tar -zvxf $OP_TAR_NAME \
        && rm -rf $OP_TAR_NAME
RUN echo "from .custom_setup_ops import *">$OP_INSTALL_DIR/__init__.py 
RUN $PYTHON -m pip install -U --force-reinstall git+https://github.com/mlperf/logging.git@2.0.0-rc1 
RUN mkdir -p $PY_PACKAGE_DIR/pybind \
        && cd $PY_PACKAGE_DIR/pybind \
        && curl -O $WHEEL_URL/$PYBIND_FUNCTION_SO_NAME  
COPY requirements.txt .
RUN $PYTHON -m pip install -r requirements.txt
RUN $PYTHON -m pip install -U --force-reinstall protobuf==3.20.1
EOF

docker build -t $DST_IMAGE \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  --build-arg no_proxy=$no_proxy \
  -f $TMP_DOCKERFILE .

rm -rf $TMP_DOCKERFILE
cd $OLD_DIR
