#!/bin//bash
cd ../pytorch
source config_G492-ID0_001x08x032.sh
export CONT=mlperfv2.0-gigabyte:ssd-20220509
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh

