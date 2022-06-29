#!/bin/bash

cd ../mxnet
source ./config_2R5500G5-Intelx8A100-SXM-80GB.sh
CONT=mlperf-H3C:unet3d DATADIR=/PATH/TO/DATADIR LOGDIR=/PATH/TO/LOGDIR sbatch -N 2 --tasks=64 --gres=gpu:8 -t 800 run.sub
