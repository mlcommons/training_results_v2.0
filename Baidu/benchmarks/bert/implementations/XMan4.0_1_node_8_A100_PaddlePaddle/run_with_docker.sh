#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

func_get_cuda_visible_devices() {
  local NGPUS=`nvidia-smi --list-gpus | wc -l` 
  local NGPUS_M1=$(($NGPUS-1))
  seq -s , 0 $NGPUS_M1
}

# Vars without defaults
: "${STAGE:?STAGE not set}"
: "${CONT:?CONT not set}"
: "${BASE_DATA_DIR:?BASE_DATA_DIR not set}"

# Vars with defaults
: "${CLEAR_CACHES:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${LOG_DIR:=$(pwd)/results}"
: "${MLPERF_MODEL_CONSTANT:=constants.BERT}"
: "${NEXP:=5}"

: "${DGXNGPU:=`nvidia-smi --list-gpus | wc -l`}"
: "${CUDA_VISIBLE_DEVICES:=`func_get_cuda_visible_devices`}"
: "${DGXSYSTEM:="DGXA100_1x8x56x1"}"
: "${CONFIG_FILE:="./config_${DGXSYSTEM}.sh"}"
: "${LOG_FILE_BASE:="${LOG_DIR}/${DATESTAMP}"}"
: "${CONT_NAME:=language_model}"
: "${NV_GPU:="${CUDA_VISIBLE_DEVICES}"}"
: "${MASTER_PORT:="29500"}"

export DGXNGPU
export CUDA_VISIBLE_DEVICES

readonly docker_image=${CONT}

# Setup directories
mkdir -p "${LOG_DIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${CONFIG_FILE} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(SEED)
_config_env+=(MASTER_PORT)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

if [[ "$STAGE" == "LOGIN" ]]; then
    docker exec -it "${_config_env[@]}" "${CONT_NAME}" bash
    exit 0
fi

# Cleanup container
cleanup_docker() {
    docker container rm -f "${CONT_NAME}" || true
}
cleanup_docker

if [[ "$STAGE" == "RUN" || "$STAGE" == "run" ]]; then
    trap 'set -eux; cleanup_docker' EXIT
fi

NVIDIA_SMI=`which nvidia-smi`

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --ulimit=nofile=1000000 \
    --name="${CONT_NAME}" \
    --env BASE_DATA_DIR=$BASE_DATA_DIR \
    -v $NVIDIA_SMI:$NVIDIA_SMI \
    -v $PWD:/workspace/bert \
    -v $BASE_DATA_DIR:$BASE_DATA_DIR \
    -w /workspace/bert \
    "${CONT}" sleep infinity

#make sure container has time to finish initialization
sleep 30
docker exec "${CONT_NAME}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
        fi

        # Run experiment
	export SEED=${SEED:-"$RANDOM"}
        docker exec "${_config_env[@]}" "${CONT_NAME}" bash "./run_and_time.sh"
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
done
