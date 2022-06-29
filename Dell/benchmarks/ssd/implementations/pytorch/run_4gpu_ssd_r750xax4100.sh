#!/bin/bash
set -x

source config_R750xax4A100-PCIE-80GB_node043.sh
#export DATASET_DIR="/mnt/data/openimages_ds/open-images-v6-mlperf"
LOGDIR="$(pwd)/results_node043}"
CONT=b89e1443c5cd ./run_with_docker_node043.sh
