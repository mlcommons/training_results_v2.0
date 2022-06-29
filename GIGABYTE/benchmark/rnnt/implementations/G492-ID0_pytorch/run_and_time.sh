#!/bin/bash
cd ../pytorch
source config_G492-ID0_1x8x192x1.sh
export CONT=mlperfv2.0-gigabyte:rnnt-20220509
export LOGDIR=/path/to/logfile
./run_with_docker.sh

