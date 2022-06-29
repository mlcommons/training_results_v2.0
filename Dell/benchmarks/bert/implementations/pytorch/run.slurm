#!/bin/bash
#SBATCH --job-name bert
#SBATCH -N 2                     # number of nodes
#SBATCH -n 8                     # total number of processes
#SBATCH -w node020,node021
#SBATCH --ntasks-per-node 4      # tasks per node
#SBATCH -t 12:00:00             # wall time
##SBATCH --exclusive             # exclusive node access
##SBATCH --mem=0                 # all mem avail
##SBATCH -p r750xa
##SBATCH --gres=gpu:4


module purge
#module avail
module load shared
module load slurm
module load shared openmpi/4.1.1
module list

GPU_TYPE=$( nvidia-smi -L | awk '{ print $4 }'| head -n 1 )

source config_${SLURM_JOB_NUM_NODES}xXE8545x4${GPU_TYPE}.sh
echo source config_${SLURM_JOB_NUM_NODES}xXE8545x4${GPU_TYPE}.sh

# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
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

#set -eux


CONT="/mnt/data/bert_20220509.sif"
#/mnt/data/mlperf/bert
: "${LOGDIR:=/home/frank/results/bert-2.0/${SLURM_JOB_NUM_NODES}XE8545-4x${GPU_TYPE}}}"
: "${NEXP:=10}"
: ${DATAPATH:="/mnt/data/mlperf/bert"}
export DATADIR="$DATAPATH/hdf5_4320_shards_varlength"
export EVALDIR="$DATAPATH/eval_varlength/"
export DATADIR_PHASE2=$DATADIR
export CHECKPOINTDIR_PHASE1="$DATAPATH/phase1"
export CHECKPOINTDIR="$CHECKPOINTDIR_PHASE1"

#######################################
# Print mount locations from file.
# Arguments:
#   arg1 - path to file containing volume mounting points
# Returns:
#   String containing comma-separated mount pairs list
#######################################
func_get_container_mounts() {
  declare -a mount_array
  readarray -t mount_array < "${1}" 
  local cont_mounts=$(envsubst <<< $(printf '%s,' "${mount_array[@]}" | sed 's/[,]*$//'))
  echo $cont_mounts
}

#######################################
# CI does not make the current directory the model directory. It is two levels up, which is different than a command line launch.
# This function looks in ${PWD} and then two levels down for a file, and updates the path if necessary.
# Arguments:
#   arg1 - expected path to file
#   arg2 - model path (e.g., language_model/pytorch/)
# Returns:
#   String containing updated (or original) path
#######################################
func_update_file_path_for_ci() {
  declare new_path
  if [ -f "${1}" ]; then
    new_path="${1}"
  else
    new_path="${2}/${1}"
  fi

  if [ ! -f "${new_path}" ]; then
    echo "File not found: ${1}"
    exit 1
  fi

  echo "${new_path}"
}

# Vars without defaults
: "${CONT:?CONT not set}"
: "${DGXSYSTEM:?DGXSYSTEM not set}"
#: "${NEXP:?NEXP not set}"

# Vars with defaults
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${API_LOGGING:=0}"
: "${CLEAR_CACHES:=1}"
#: "${CONT_FILE:=/lustre/fsw/containers/${SLURM_JOBID}_$(basename ${CONT}).squashfs}"
: "${CONT_NAME:=language_model}"
: "${CONTAINER_PRELOAD_LUSTRE:=0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
#: "${LOGDIR:=./results}"
: "${NSYSCMD:=""}"
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${WORK_DIR:=/workspace/bert}"

if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" nsys profile --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/bert_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}.qdrep "
fi

if [ ${TIME_TAGS} -gt 0 ]; then
    LOG_BASE="${LOG_BASE}_mllog"
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
    LOG_BASE="${LOG_BASE}_nsys"
fi

LOG_BASE="bert_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"
readonly LOG_FILE_BASE="${LOGDIR}/${LOG_BASE}"

srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 mkdir -p "${CHECKPOINTDIR}"

if [[ $CONTAINER_PRELOAD_LUSTRE -gt 0 ]]; then
    CONT_FILE="/lustre/fsw/containers/${SLURM_JOBID}_$(basename ${CONT}).squashfs"
    # Prepull container to LUSTRE
    srun --ntasks=1 enroot import --output ${CONT_FILE} docker://${CONT}
else
    CONT_FILE=${CONT}
fi

#echo "CI directory structure\n"
#echo $(ls)

CONT_MOUNTS=$(func_get_container_mounts $(func_update_file_path_for_ci mounts.txt ${PWD}/language_model/pytorch))

if [ "${API_LOGGING}" -eq 1 ]; then
    CONT_MOUNTS="${CONT_MOUNTS},${API_LOG_DIR}:/logs"
fi


# Setup directories
#mkdir -p "${LOGDIR}"
#srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
#srun --ntasks="$((SLURM_JOB_NUM_NODES))" --container-image="${CONT_FILE}" --container-name="${CONT_NAME}" true

# Run NCCL test (700 MB FP16 allreduce)
#srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#        --container-image=${CONT_FILE} \
#        all_reduce_perf_mpi -b 85M -e 680M -f 2 -d half


#export NCCL_DEBUG=INFO
#export NCCL_TOPO_FILE="/workspace/bert/dgxa100_nic_affinity.xml"
export NCCL_TOPO_FILE="/workspace/bert/xe8545_nic_affinity.xml"

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        hosts=$(scontrol show hostname |tr "\n" " ")

        echo "hosts=$hosts"
        #for node_id in `seq 0 $(($NUM_NODES-1))`; do
        for node in $hosts; do

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
#            srun -N 1 -n 1 -w $node bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
            srun -N 1 -n 1 -w $node mpirun --allow-run-as-root -np 1 singularity exec -B $PWD:/workspace/bert --pwd /workspace/bert $CONT python -c "
import mlperf_logger
mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)"

        fi
	done

        # Run experiment
#	    srun -l --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" --container-name="${CONT_NAME}" --container-mounts="${CONT_MOUNTS}" --container-workdir=${WORK_DIR} "run_and_time.sh"
#    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
        #MPIRUN="mpirun --allow-run-as-root --bind-to none --report-bindings -np $SLURM_NTASKS --map-by node:pe=4"
#        MPIRUN="mpirun --allow-run-as-root --bind-to none --report-bindings -np $SLURM_NTASKS --map-by node:pe=4"
        MPIRUN="mpirun --allow-run-as-root --bind-to none --report-bindings -np $SLURM_NTASKS"
#MPIRUN="mpirun --allow-run-as-root --bind-to none  --report-bindings --host $(scontrol show hostname |tr "\n" ",") -np $SLURM_NTASKS"
#MPIRUN="mpirun --allow-run-as-root --bind-to none  --report-bindings --host node020:4,node021:4 -np $SLURM_NTASKS"

#echo CONT_MOUNTS="${CONT_MOUNTS}"
#export SINGULARITY_BIND="/opt,/data:/mnt"
#	 srun -l --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" singularity exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert \

	# ${MPIRUN} singularity exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert \
#                --pwd /workspace/bert \
#                $CONT bash ./run_and_time_multi.sh
#                ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"

	 #${MPIRUN} singularity exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert \
	 #srun -l --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=4 singularity exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert \
#	 srun -l --ntasks=8  --ntasks-per-node=4 singularity exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert \
	 srun -l --ntasks=${SLURM_NTASKS}  --ntasks-per-node=${DGXNGPU} singularity exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert \
                --pwd /workspace/bert \
                $CONT  ./run_and_time_multi.sh
                ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"

                #$CONT  ./run.slurm.sh
done

## Cleanup
#if [[  $CONTAINER_PRELOAD_LUSTRE -gt 0 ]]; then
#    srun --ntasks=1 rm ${CONT_FILE}
#fi
