## DL params
export BATCH_SIZE=55296
export DGXNGPU=4

export CONFIG="dgx_a100.py"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=5
export WALLTIME=$(( 5 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))
export OMPI_MCA_btl="^openib"
export MOUNTS=/raid:/raid
export CUDA_DEVICE_MAX_CONNECTIONS=3
