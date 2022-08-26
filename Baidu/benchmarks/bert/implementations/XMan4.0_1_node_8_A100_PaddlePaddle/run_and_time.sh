# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

ldconfig

export PYTHON=python3.8
export PADDLE_TRAINERS_NUM=`$PYTHON -c "import paddle; print(paddle.device.cuda.device_count())"`
default_addr="127.0.0.1"
default_port="60001"

for i in `seq 1 10`; do
  for port_idx in `seq 0 $(($PADDLE_TRAINERS_NUM-1))`;
  do
    port=$(($default_port+$port_idx))
    ((lsof -i:$port | tail -n +2 | awk '{print $2}' | xargs kill -9) || true) >/dev/null 2>&1
  done
done

set -ex

export SEED=${SEED:-"$RANDOM"}
export PADDLE_TRAINER_ENDPOINTS=`$PYTHON -c "print(\",\".join([\"$default_addr:\" + str($default_port + i) for i in range($PADDLE_TRAINERS_NUM)]))"`

CMD="bash run_benchmark.sh"

echo $BASE_DATA_DIR

bash kill_grep.sh $PYTHON || true

num_process=$(($PADDLE_TRAINERS_NUM*2))
echo $num_process

if [[ $num_process -gt 1 ]]; then
  ORTERUN=`which orterun`
  mpirun="$ORTERUN --allow-run-as-root \
     -np $num_process \
     -mca btl_tcp_if_exclude docker0,lo,matrixdummy0,matrix0 \
     --bind-to none -x PADDLE_TRAINERS_NUM \
     -x PADDLE_TRAINER_ENDPOINTS -x LD_LIBRARY_PATH -x SEED -x PYTHON \
     -x CUDA_VISIBLE_DEVICES -x SEED"
else
  mpirun=""
fi

$mpirun $CMD

# bash kill_grep.sh run_and_time || true
# bash kill_grep.sh python || true
