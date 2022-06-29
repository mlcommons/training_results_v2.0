#!/bin/bash 


source config_R750xax4A100PCIE80GB.sh

#nvcr.io/nvdlfwea/mlperfv20/resnet:20220502.mxnet.cpubindoff
CONT=41da595508ee   ./run_with_docker.sh
