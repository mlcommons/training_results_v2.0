#!bin/bash

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

set -x

run_test() {
  cd "$1"
  echo "begin test: "
  for i in `seq $2`
  do
    bash run_and_time.sh;
    dst_dir=log_8_allopt_xxx_$i
    rm -rf $dst_dir;
    mv log_8 $dst_dir;
    sleep 10
  done
  cd -
}

# TODO: change the following directory to your codebase path    
PADDLE_DIR="/data2/zengjinle/mlperf/v2.0/benchmarks/bert/implementations"
NV_DIR="/data2/zengjinle/submission_training_1.1/NVIDIA/benchmarks/bert/implementations/pytorch"

run_test $PADDLE_DIR 10
run_test $NV_DIR 20
