#!/bin/bash

cd ../hugectr
source ./config_R5300G5x8A100-PCIE-80GB.sh
CONT=mlperf-H3C:dlrm DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh