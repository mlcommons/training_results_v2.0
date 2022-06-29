#!/bin/bash

cd ../mxnet
source ./config_R5500G5-AMDx8A100-SXM-80GB.sh
CONT=mlperf-H3C:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh