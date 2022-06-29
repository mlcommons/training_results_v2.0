#!/bin/bash

cd ../pytorch
source config_NF5688M6.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DGXSYSTEM="NF5688M6" CONT=mlperf-inspur:bert ./run_with_docker.sh
