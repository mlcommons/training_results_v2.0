#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded

if [ "$(docker images | grep mlperfv20 | grep resnet)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.resnet.bz2
	# Wait a little
	sleep 30
fi


# Copy dataset on local NVME
mkdir -p /beeond/preprocess/
rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/ /beeond/preprocess/

export CONT=nvcr.io/nvdlfwea/mlperfv20/resnet:20220509.mxnet 

source ${curDir}/mxnet/config_675D.sh

export DATADIR=/beeond/preprocess 
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

cd mxnet
./run_with_docker_HPE.sh
