#!/bin/bash

source config_DGXA100.sh
NEXP=5 CONT=nvcr.io/nvdlfwea/mlperfv20/resnet:20220509.mxnet DATADIR=/mnt/data4/work/forMXNet_no_resize LOGDIR=LOG DGXSYSTEM=DGXA100 PULL=0 ./run_with_docker.sh
