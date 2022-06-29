#!/bin/bash

cd ../hugectr
source ./config_R5300G5x4A100-SXM-80GB.sh
CONT=mlperf-H3C:dlrm DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh