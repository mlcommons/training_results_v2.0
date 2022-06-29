#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded
if [ "$(docker images | grep mlperfv20 | grep maskrcnn)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.maskrcnn.bz2
	# Wait a little
	sleep 10
fi

export CONT=nvcr.io/nvdlfwea/mlperfv20/maskrcnn:20220509.pytorch

source ${curDir}/pytorch/config_675D.sh

export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/ # /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017/
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}


cd pytorch
./run_with_docker_HPE.sh
