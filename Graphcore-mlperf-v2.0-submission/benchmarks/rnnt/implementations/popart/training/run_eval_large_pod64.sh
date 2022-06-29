#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

REPLICAS=$1
INSTANCES=$2

PARTITION=$3
VIPU_SERVER_HOST=$4
NETMASK=$5

MODEL_DIR=$6
NUM_EPOCHS=$7
START_EVAL_EPOCH=$8
WER_TARGET=$9

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/LibriSpeech/
export EXECUTABLE_CACHE_EVAL=/localdata/$USER/executable_cache_transformer_transducer_eval_large
export POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout":"900"}'
export POPLAR_RUNTIME_OPTIONS='{"streamCallbacks.maxLookahead":"unlimited"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include "$NETMASK" --mca btl_tcp_if_include "$NETMASK"' \
    --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL -x POPART_LOG_LEVEL -x RNNT_LOG_LEVEL -x POPLAR_ENGINE_OPTIONS -x POPLAR_RUNTIME_OPTIONS' \
    --update-partition=yes --reset-partition=no --vipu-server-timeout 600 \
    --numa-aware 1 --vipu-server-host "$VIPU_SERVER_HOST" \
    --vipu-partition="$PARTITION" "

EVAL=" poprun \
    -vv $MPI_SETTINGS \
    --executable-cache-path "$EXECUTABLE_CACHE_EVAL" \
    --num-instances $INSTANCES --num-replicas $REPLICAS --ipus-per-replica 2 \
    python3 transducer_validation.py --model-conf-file configs/transducer-large-1023sp.yaml \
	--model-dir "$MODEL_DIR" \
	--data-dir "$DATA_DIR" \
	--enable-half-partials --enable-lstm-half-partials \
    --replication-factor $REPLICAS \
	--validation-epoch-span $START_EVAL_EPOCH $NUM_EPOCHS --wer-target $WER_TARGET "
	
eval $EVAL