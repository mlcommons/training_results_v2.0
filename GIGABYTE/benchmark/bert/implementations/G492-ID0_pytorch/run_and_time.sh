#!/bin/bash
cd ../pytorch
source config_G492-ID0_1x8x56x1.sh
export CONT=mlperfv2.0-gigabyte:bert-20220509
export LOGDIR=/path/to/logfile
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./run_with_docker.sh

