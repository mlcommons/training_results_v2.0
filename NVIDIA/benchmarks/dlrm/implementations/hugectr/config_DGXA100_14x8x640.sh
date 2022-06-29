## DL params
export BATCH_SIZE=71680
export DGXNGPU=8

export CONFIG="dgx_a100_14x8x640.py"

## System run parms
export DGXNNODES=14
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_BASE=$(( 10 + 60 * ${API_LOGGING:-0} ))
WALLTIME_MINUTES=5
export WALLTIME=$(( WALLTIME_BASE + (${NEXP:-1} * WALLTIME_MINUTES) ))
export OMPI_MCA_btl="^openib"
export MOUNTS=/raid:/raid
export CUDA_DEVICE_MAX_CONNECTIONS=3

export SBATCH_NETWORK=sharp
export SBATCH_OTHER_PARAMS="--switches 1@00:10:00"
