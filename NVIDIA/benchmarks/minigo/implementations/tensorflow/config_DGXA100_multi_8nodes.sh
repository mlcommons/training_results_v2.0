#
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_multi.sh
export NUM_GPUS_TRAIN=16
# System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=03:59:00

#
export NUM_ITERATIONS=80
