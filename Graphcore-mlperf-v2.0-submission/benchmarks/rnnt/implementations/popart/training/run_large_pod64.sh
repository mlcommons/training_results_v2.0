#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

REPLICAS=$1
INSTANCES=$2

HOSTS=$3
PARTITION=$4
VIPU_SERVER_HOST=$5
NETMASK=$6

MODEL_DIR=$7
NUM_EPOCHS=$9

EXECUTABLE_CACHE_DIR=${10}

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/LibriSpeech/
export EXECUTABLE_CACHE=${EXECUTABLE_CACHE_DIR}/executable_cache
export POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout":"900"}'
export POPLAR_RUNTIME_OPTIONS='{"streamCallbacks.maxLookahead":"unlimited"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include "$NETMASK" --mca btl_tcp_if_include "$NETMASK"' \
    --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL -x POPART_LOG_LEVEL -x RNNT_LOG_LEVEL -x POPLAR_ENGINE_OPTIONS -x POPLAR_RUNTIME_OPTIONS -x SHARED_EXECUTABLE_CACHE=1' \
    --update-partition=yes --reset-partition=no --vipu-server-timeout 600 \
    --ipus-per-replica 2 --numa-aware 1 --vipu-server-host "$VIPU_SERVER_HOST" \
    --vipu-partition="$PARTITION" \
    --executable-cache-path "$EXECUTABLE_CACHE" "


TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python3 transducer_train.py --model-conf-file configs/transducer-large-1023sp.yaml \
	--model-dir "$MODEL_DIR" \
	--data-dir "$DATA_DIR" \
	--enable-half-partials --enable-lstm-half-partials --enable-stochastic-rounding --joint-net-custom-op \
	--replication-factor "$REPLICAS" --batch-size "$BATCH_SIZE" --gradient-accumulation-factor "$GRAD_ACCUM" --num-epochs "$NUM_EPOCHS" "
	
echo "Running Transformer-Transducer training:"
echo $TRAIN
eval $TRAIN