#!/bin/bash

cd ../pytorch
source ./config_R5300G5x4A100-SXM-80GB.sh
CONT=mlperf-H3C:maskrcnn DATADIR=/PATH/TO/DATADIR PKLDIR=/PATH/TO/PKLDIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh