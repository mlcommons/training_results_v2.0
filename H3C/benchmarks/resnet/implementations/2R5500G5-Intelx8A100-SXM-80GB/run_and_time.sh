#!/bin/bash

cd ../mxnet
source ./config_2R5500G5-Intelx8A100-SXM-80GB_common.sh
CONT=mlperf-H3C:resnet DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR  sbatch -N 2 --tasks=64 --gres=gpu:8 -t 400 run.sub
