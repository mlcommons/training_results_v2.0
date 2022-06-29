#!/bin/bash

NUM_WORKERS_PER_HLS=8
HLS_TYPE=HLS1
./launch_keras_resnet_hvd.sh --cpu-pin cpu
