#!/bin/bash
#SBATCH --job-name maskrcnn
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


#set -euxo pipefail
set -x 

CONT="/mnt/data/maskrcnn_20220509.sif"
: "${LOGDIR:=/home/frank/results/maskrcnn-2.0/${SLURM_JOB_NUM_NODES}XE8545-4x${GPU_TYPE}}}"
: "${NEXP:=5}"
: "${DATADIR:=/mnt/data/coco2017}"
#export DATADIR="/dev/shm/coco2017"
export DATADIR="/mnt/data/coco2017"

mkdir -p $LOGDIR

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# pickled annotation path
case $DGXSYSTEM in
DGX1)
    PKLPATH="/raid/datasets/coco_pkl_annotations" ;;
*)
    PKLPATH="/mnt/data/coco2017/pkl_coco" ;;
esac

# Vars with defaults
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${PKLDIR:=$PKLPATH}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir

LOGBASE="${DATESTAMP}"
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
EPOCH_PROF=${EPOCH_PROF:-0}

SPREFIX="maskrcnn_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"

if [ ${TIME_TAGS} -gt 0 ]; then
    LOGBASE="${SPREFIX}_mllog"
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nsys"
    else
        LOGBASE="${SPREFIX}_nsys"
    fi
fi
if [ ${SYNTH_DATA} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_synth"
    else
        LOGBASE="${SPREFIX}_synth"
    fi
fi
if [ ${EPOCH_PROF} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_epoch"
    else
        LOGBASE="${SPREFIX}_epoch"
    fi
fi

# Other vars
readonly _logfile_base="${LOGDIR}/${LOGBASE}"
readonly _cont_name=object_detection
#_cont_mounts="${DATADIR}:/data,${PKLDIR}:/pkl_coco,${LOGDIR}:/results"
_cont_mounts="${DATADIR}:/coco,${PKLDIR}:/pkl_coco,${LOGDIR}:/results"
if [ "${API_LOGGING:-}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"
fi

# MLPerf vars
MLPERF_HOST_OS=$(srun -N1 -n1 bash <<EOF
    source /etc/os-release
    source /etc/dgx-release || true
    echo "\${PRETTY_NAME} / \${DGX_PRETTY_NAME:-???} \${DGX_OTA_VERSION:-\${DGX_SWBUILD_VERSION:-???}}"
EOF
)
export MLPERF_HOST_OS

# Setup directories
#mkdir -p "${LOGDIR}"
#srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
#srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true

echo "NCCL_TEST = ${NCCL_TEST}"
if [[ ${NCCL_TEST} -eq 1 ]]; then
    (srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
         --container-name="${_cont_name}" all_reduce_perf_mpi -b 82.6M -e 82.6M -d half \
)  |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

fi

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}"

        hosts=$(scontrol show hostname |tr "\n" " ")
        echo "hosts=$hosts"
        #for node_id in `seq 0 $(($NUM_NODES-1))`; do
        for node in $hosts; do



        # Print system info
        #srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
        #srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
	srun -N 1 -n 1 -w $node mpirun --allow-run-as-root -np 1 singularity exec -B $PWD:/workspace/object_detection --pwd /workspace/object_detection $CONT python -c "
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.mlperf_logger import mlperf_submission_log
mlperf_submission_log(constants.MASKRCNN)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
#            srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
#            srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
	     srun -N 1 -n 1 -w $node bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
	     srun -N 1 -n 1 -w $node mpirun --allow-run-as-root -np 1 singularity exec -B $PWD:/workspace/object_detection --pwd /workspace/object_detection $CONT python -c "
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.mlperf_logger import log_event
log_event(key=constants.CACHE_CLEAR, value=True, stack_offset=1)"
        fi
	
	done

        # Run experiment
#        srun --mpi=none --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#            --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
#            ./run_and_time.sh
         srun -l --ntasks=${SLURM_NTASKS}  --ntasks-per-node=${DGXNGPU} singularity exec --nv -B "${_cont_mounts}" -B $PWD:/workspace/object_detection \
                --pwd /workspace/object_detection \
                $CONT  ./run_and_time_multi.sh

    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
