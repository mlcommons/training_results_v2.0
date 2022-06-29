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

DIR=$(readlink -f `dirname "$0"`)
g++ "$DIR/nccl.cc" -std=c++17 -fPIC -shared -o "$DIR/libnccl_wrapper.so" -I/usr/local/cuda/include -ldl -lnccl

(cd "$DIR" && rm -rf build && python3.8 setup.py install --force)

echo "Set the following env before run:"
echo ""
echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'$DIR" LD_PRELOAD=$DIR/libnccl_wrapper.so"
echo ""
