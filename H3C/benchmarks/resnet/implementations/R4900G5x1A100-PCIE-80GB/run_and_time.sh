#!/bin/bash

cd ../mxnet
source ./config_R4900G5x1A100-PCIE-80GB.sh
CONT=mlperf-H3C:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh