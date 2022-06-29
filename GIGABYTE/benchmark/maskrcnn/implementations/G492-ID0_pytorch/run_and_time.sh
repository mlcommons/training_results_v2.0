#!/bin/bash
cd ../pytorch
source config_G492-ID0.sh
export CONT=mlperfv2.0-gigabyte:maskrcnn-20220509
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh
