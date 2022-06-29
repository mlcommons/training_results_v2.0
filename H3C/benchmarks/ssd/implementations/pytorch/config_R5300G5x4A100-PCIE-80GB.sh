#!/usr/bin/env bash

## DL params
export BATCHSIZE=${BATCHSIZE:-32}
export NUMEPOCHS=${NUMEPOCHS:-8}
export DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6"}
export LR=${LR:-0.0001}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-head-fusion --disable-ddp-broadcast-buffers --fp16-allreduce --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --skip-metric-loss --cuda-graphs-syn'}

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=160
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
