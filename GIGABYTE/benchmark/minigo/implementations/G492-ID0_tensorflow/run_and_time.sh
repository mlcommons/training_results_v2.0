#!/bin//bash
cd ../tensorflow
source config_G492-ID0.sh
export CONT=mlperfv2.0-gigabyte:minigo-20220509
export LOGDIR=/path/to/logfile
./run_with_docker.sh

