#!/bin/bash

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

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set +x
set -e

# Only rank print
[ "${SLURM_LOCALID-0}" -ne 0 ] && set +x


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
[ "${DEBUG}" = "1" ] && set -x
LR=${LR:-0.0001}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
BATCHSIZE=${BATCHSIZE:-2}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6"}
export TORCH_HOME=${TORCH_HOME:-"/torch-home"}
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}

# run benchmark
echo "running benchmark"
if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" nsys profile --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/single_stage_detector_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCH_SIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}.nsys-rep "
else
 NSYSCMD=""
fi

if [ ${SYNTH_DATA} -gt 0 ]; then
${EXTRA_PARAMS}="${EXTRA_PARAMS} --syn-dataset"
EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--dali//')
fi

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
    cluster=''
    if [[ "${DGXSYSTEM}" == DGX2* ]]; then
        cluster='circe'
    fi
    if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
        cluster='selene'
    fi
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' "--cluster=${cluster}" '--ib=single' '--' ${NSYSCMD} 'python' '-u' )
  else
    CMD=( ${NSYSCMD} 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  # CMD=( "${NSYSCMD}" "python" "-u" "-m" "bind_launch" "--nsockets_per_node=${DGXNSOCKET}" \
  #       "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
  # [ "$MEMBIND" = false ] &&  CMD+=( "--no_membind" )
  # TODO: Replace below CMD with NSYSCMD..., but make sure NSYSCMD is not an empty string
  CMD=( "python" )
fi


if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

PARAMS=(
      --lr                      "${LR}"
      --batch-size              "${BATCHSIZE}"
      --eval-batch-size         "${EVALBATCHSIZE}"
      --epochs                  "${NUMEPOCHS}"
      --print-freq              "${LOG_INTERVAL}"
      --dataset-path            "${DATASET_DIR}"
      --warmup-epochs           "${WARMUP_EPOCHS}"
)

# run training
${LOGGER:-} "${CMD[@]}" train.py "${PARAMS[@]}" ${EXTRA_PARAMS} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
