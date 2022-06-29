#!/bin/bash

cd ../mxnet
source ./config_R5500G5-Intelx8A100-SXM-80GB-400W.sh
CONT=mlperf-H3C:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  ./run_with_docker.sh