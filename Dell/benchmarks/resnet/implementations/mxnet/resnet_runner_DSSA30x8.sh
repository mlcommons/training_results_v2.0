#!/bin/bash 

#export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
source config_DSS8440_A30.sh
NEXP=5
#CONT=88f3cf4cbc4c DATADIR=/mnt/data3/ilsvrc12_passthrough/ LOGDIR=${PWD}/results/ ./run_with_docker.sh 

#nvcr.io/nvdlfwea/mlperfv20/resnet:20220502.mxnet.cpubindoff
CONT=6cf555594a21 DATADIR=/dev/shm/ilsvrc12_passthrough/ LOGDIR=${PWD}/results/ ./run_with_docker.sh 
