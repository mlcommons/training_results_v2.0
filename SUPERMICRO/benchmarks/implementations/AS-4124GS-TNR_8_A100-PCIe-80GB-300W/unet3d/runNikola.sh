#!/bin/bash

export shardsdir='/home/smci/MLPerfTraining2.0/unet3d/data/data'
echo $shardsdir

export chkdir='/home/smci/MLPerfTraining2.0/unet3d/data/logs'
echo $chkdir


#source ./config_DGXA100_1x4x56x2.sh
#sudo nvidia-smi -pm 1
#sudo nvidia-smi -lgc 1512,1512
#

source config_DGXA100_1x8x7.sh
#source ./config_DGXA100_1x8x56x1.sh
CONT=nvcr.io/nvdlfwea/mlperfv20/unet3d:20220425.mxnet DATADIR=$shardsdir LOGDIR=$chkdir ./run_with_docker.sh

docker stop image_segmentation
docker rm image_segmentation

