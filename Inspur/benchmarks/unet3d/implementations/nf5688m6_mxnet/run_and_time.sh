#!/bin/bash

cd ../mxnet
source config_NF5688M6.sh
DGXSYSTEM="NF5688M6" CONT=mlperf-inspur:unet3d DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile ./run_with_docker.sh
