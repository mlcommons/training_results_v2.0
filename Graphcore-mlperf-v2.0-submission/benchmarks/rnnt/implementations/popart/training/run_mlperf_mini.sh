#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

NUM_RUN=$1
NUM_EPOCHS=$2
START_EVAL_EPOCH=$3
WER_TARGET=$4

export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/LibriSpeech/
export MODEL_DIR=/localdata/$USER/TRANSFORMER_TRANSDUCER_CHECKPOINTS_TINY
export EXECUTABLE_CACHE_TRAIN=/localdata/$USER/executable_cache_transformer_transducer_train_mini
export EXECUTABLE_CACHE_EVAL=/localdata/$USER/executable_cache_transformer_transducer_eval_mini

TRAIN="python3 transducer_train.py --model-conf-file configs/transducer-mini.yaml \
    --model-dir "$MODEL_DIR" \
	--data-dir "$DATA_DIR" \
    --enable-half-partials --enable-lstm-half-partials --enable-stochastic-rounding --joint-net-custom-op \
    --executable-cache-path "$EXECUTABLE_CACHE_TRAIN" \
    --num-epochs $NUM_EPOCHS \
    --mlperf-log-path results/ipu-POD64-popart/rnnt/result_"$NUM_RUN".txt "

EVAL="python3 transducer_validation.py --model-conf-file configs/transducer-mini.yaml \
    --model-dir "$MODEL_DIR" \
	--data-dir "$DATA_DIR" \
    --enable-half-partials --enable-lstm-half-partials \
    --executable-cache-path "$EXECUTABLE_CACHE_EVAL" \
    --validation-epoch-span $START_EVAL_EPOCH $NUM_EPOCHS\
    --wer-target $WER_TARGET \
    --mlperf-log-path results/ipu-POD64-popart/rnnt/result_"$NUM_RUN".txt "

$TRAIN
$EVAL