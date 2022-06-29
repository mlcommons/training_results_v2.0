#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source config_DSS8440x8xA30.sh
export NEXP=10
CONT=5dc7e69684af ./run_with_docker.sh
