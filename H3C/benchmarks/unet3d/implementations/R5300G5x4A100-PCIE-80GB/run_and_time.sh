#!/bin/bash

cd ../mxnet
source ./config_R5300G5x4A100-PCIE-80GB.sh
CONT=mlperf-H3C:unet3d DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR bash ./run_with_docker.sh