#!/bin/bash

cd ../mxnet
source ./config_2R5500G5-Intelx8A100-SXM-80GB.sh
CONT=mlperf-H3C:ssd DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR TORCH_HOME=/PATH/TO/TORCH_HOME  sbatch -N 2 --tasks=64 --gres=gpu:8 -t 500 run.sub
