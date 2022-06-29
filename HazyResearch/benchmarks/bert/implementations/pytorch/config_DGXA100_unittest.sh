export UNITTEST=1
export EXTRA_PARAMS=""
export BATCHSIZE=1

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:15:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

