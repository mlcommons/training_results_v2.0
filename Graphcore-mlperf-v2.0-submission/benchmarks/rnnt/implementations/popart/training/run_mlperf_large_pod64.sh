#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

RUN=$1
INSTANCES=$2

HOSTS=$3
PARTITION=$4
VIPU_SERVER_HOST=$5
NETMASK=$6

NUM_EPOCHS=$7
START_EVAL_EPOCH=$8
WER_TARGET=$9
EXECUTABLE_CACHE_DIR=${10}

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/LibriSpeech/
export MODEL_DIR=/localdata/TRANSFORMER_TRANSDUCER_CHECKPOINTS_MLPERF_$RUN
export EXECUTABLE_CACHE_TRAIN=${EXECUTABLE_CACHE_DIR}/executable_cache_transformer_transducer_train_large
export EXECUTABLE_CACHE_EVAL=${EXECUTABLE_CACHE_DIR}/executable_cache_transformer_transducer_eval_large
export POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout":"900"}'
export POPLAR_RUNTIME_OPTIONS='{"streamCallbacks.maxLookahead":"unlimited"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include "$NETMASK" --mca btl_tcp_if_include "$NETMASK"' \
    --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL -x POPART_LOG_LEVEL -x RNNT_LOG_LEVEL -x POPLAR_ENGINE_OPTIONS -x POPLAR_RUNTIME_OPTIONS -x SHARED_EXECUTABLE_CACHE=1' \
    --update-partition=yes --reset-partition=no --vipu-server-timeout 600 \
    --numa-aware 1 --vipu-server-host "$VIPU_SERVER_HOST" \
    --vipu-partition="$PARTITION" "

TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --executable-cache-path "$EXECUTABLE_CACHE_TRAIN" \
    --num-instances "$INSTANCES" --num-replicas 32 --ipus-per-replica 2 \
    python3 transducer_train.py --model-conf-file configs/transducer-large-1023sp.yaml \
	--model-dir "$MODEL_DIR" \
	--data-dir "$DATA_DIR" \
	--enable-half-partials --enable-lstm-half-partials --enable-stochastic-rounding --joint-net-custom-op \
	--replication-factor 32 --batch-size 64 --gradient-accumulation-factor 16 --num-epochs "$NUM_EPOCHS" \
    --mlperf-log-path results/ipu-POD64-popart/rnnt/result_"$RUN".txt "

EVAL=" poprun \
    -vv $MPI_SETTINGS \
    --executable-cache-path "$EXECUTABLE_CACHE_EVAL" \
    --num-instances 16 --num-replicas 16 --ipus-per-replica 2 \
    python3 transducer_validation.py --model-conf-file configs/transducer-large-1023sp.yaml \
	--model-dir "$MODEL_DIR" \
	--data-dir "$DATA_DIR" \
	--enable-half-partials --enable-lstm-half-partials \
    --replication-factor 16 \
	--validation-epoch-span $START_EVAL_EPOCH $NUM_EPOCHS --wer-target $WER_TARGET \
    --mlperf-log-path results/ipu-POD64-popart/rnnt/result_"$RUN".txt "
	
echo "Running Transformer-Transducer training:"
eval $TRAIN
echo "Running Transformer-Transducer evaluation:"
eval $EVAL