#!/bin/bash

cd ../mxnet
source ./config_R5300G5x4A100-PCIE-80GB.sh
CONT=mlperf-H3C:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh