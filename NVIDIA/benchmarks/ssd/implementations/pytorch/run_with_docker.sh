#!/bin/bash

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${TORCH_HOME:=./torch-home}"
: "${CONT_NAME:=single_stage_detector}"
# ci automagically sets this correctly on Selene
: "${DATADIR:=/raid/datasets/openimages/open-images-v6}"
: "${LOGDIR:=$(pwd)/results}"
# Logging
LOG_BASE="ssd_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"
readonly LOG_FILE_BASE="${LOGDIR}/${LOG_BASE}"
# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
# Mount points
CONT_MOUNTS=(
    "--volume=${DATADIR}:/datasets/open-images-v6"
    "--volume=${LOGDIR}:/results"
    "--volume=${TORCH_HOME}:/torch-home"
)
# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME} / ${DGX_PRETTY_NAME:-???} ${DGX_OTA_VERSION:-${DGX_SWBUILD_VERSION:-???}}"
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${CONT_NAME}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
if [ -z "${NV_GPU-}" ]; then
  readonly _docker_gpu_args="--gpus all"
else
  readonly _docker_gpu_args='--gpus="'device=${NV_GPU}'" -e NVIDIA_VISIBLE_DEVICES='"${NV_GPU}"
fi

docker run ${_docker_gpu_args} --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${CONT_NAME}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
#make sure container has time to finish initialization
sleep 30
docker exec -it "${CONT_NAME}" true

readonly TORCH_RUN="python -m torch.distributed.run --standalone --no_python"

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (

        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${CONT_NAME}" python -c "
from mlperf_logging import mllog
from mlperf_logging.mllog.constants import CACHE_CLEAR
mllogger = mllog.get_mllogger()
mllogger.event(key=CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        docker exec -it "${_config_env[@]}" "${CONT_NAME}" \
               ${TORCH_RUN} --nproc_per_node=${DGXNGPU} ./run_and_time.sh
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
done
