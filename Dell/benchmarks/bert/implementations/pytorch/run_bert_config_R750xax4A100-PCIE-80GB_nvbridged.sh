#!/bin/bash

source config_R750xax4A100-PCIE-80GB.sh
export NEXP=10
# nvcr.io/nvdlfwea/mlperfv20/bert:20220502.pytorch.cpubindingoff
CONT=1a8f9fd11667 ./run_with_docker.sh 
