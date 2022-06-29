#!/bin/bash
cd ../mxnet
source config_G492-ID0_1x8x7.sh
export CONT=mlperfv2.0-gigabyte:unet3d-20220509
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh

