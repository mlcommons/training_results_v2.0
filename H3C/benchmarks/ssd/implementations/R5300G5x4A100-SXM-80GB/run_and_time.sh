#!/bin/bash

cd ../mxnet
source ./config_R5300G5x4A100-SXM-80GB.sh
CONT=mlperf-H3C:ssd DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR TORCH_HOME=/PATH/TO/TORCH_HOME bash ./run_with_docker.sh