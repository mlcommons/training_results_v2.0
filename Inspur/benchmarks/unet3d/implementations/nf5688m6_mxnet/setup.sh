#!/bin/bash

cd ../mxnet
docker build --pull -t mlperf-inspur:unet3d .
