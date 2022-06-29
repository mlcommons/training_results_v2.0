#!/bin/bash

TAG=$1
IMG=gitlab-master.nvidia.com:5005/dl/mlperf/training/maskrcnn:$1
DOCKER_BUILDKIT=1 docker build --pull --no-cache . --rm -t $IMG
docker push $IMG
