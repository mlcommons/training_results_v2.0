#!/bin/bash

# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

[ "${DEBUG}" = "1" ] && set -x

# Vars without defaults
: "${BATCHSIZE:?BATCHSIZE not set}"
: "${GRADIENT_STEPS:?GRADIENT_STEPS not set}"
: "${LR:?LR not set}"
: "${MAX_STEPS:?MAX_STEPS not set}"
: "${PHASE:?PHASE not set}"

# Vars with defaults
: "${LOCAL_RANK:=${SLURM_LOCALID}}"
: "${LOGGER:=""}"
: "${MULTI_NODE:=''}"
: "${OMPI_COMM_WORLD_LOCAL_RANK:=""}"
: "${SEED:=$RANDOM}"
: "${SLURM_JOB_ID:=$RANDOM}"
: "${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK}}"
: "${SLURM_NODEID:=0}"
: "${SLURM_NTASKS_PER_NODE:=$DGXNGPU}"
: "${THROUGHPUT_RUN:=""}"
: "${UNITTEST:=0}"

: "${EVAL_ITER_SAMPLES:=100000}"
: "${EVAL_ITER_START_SAMPLES:=100000}"
: "${MAX_SAMPLES_TERMINATION:=14000000}"
: "${OPT_LAMB_BETA_1:=0.9}"
: "${OPT_LAMB_BETA_2:=0.999}"
: "${START_WARMUP_STEP:=0.0}"
: "${TARGET_MLM_ACCURACY:=0.720}"
: "${USE_DDP:=0}"
: "${WARMUP_PROPORTION:=0.0}"
: "${WARMUP_STEPS:=0.0}"
: "${WEIGHT_DECAY_RATE:=0.01}"
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${NSYS_DELAY:=10}"
: "${NSYS_DUR:=5}"
: "${NCCL_TEST:-0}"
: "${EPOCH_PROF:-0}"
echo "$TIME_TAGS"
echo "$NVTX_FLAG"
echo "$NCCL_TEST"
echo "$EPOCH_PROF"
echo "$SYNTH_DATA"

#if [[ {$EXTRA_PARAMS} == *"use_cuda_graph"* ]]; then
# if [ ${TIME_TAGS} -gt 0 ]; then
#  TIME_TAGS=0
#  NVTX_FLAG=1
#  echo "Unset TIME_TAGS and set NVTX_FLAG because cuda graph is on"
# fi
#fi

echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

if [ ! -z "$THROUGHPUT_RUN" ]
then
  MAX_STEPS=4
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
  NSYSCMD=" nsys profile --delay ${NSYS_DELAY} --duration ${NSYS_DUR} --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/language_model_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}.nsys-rep "
fi

PHASE1="\
    --train_batch_size=${BATCHSIZE} \
    --learning_rate=${LR} \
    --warmup_proportion=${WARMUP_PROPORTION} \
    --max_steps=7038 \
    --num_steps_per_checkpoint=2500 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --input_dir=/workspace/data \
    "
PHASE2="\
    --train_batch_size=${BATCHSIZE} \
    --learning_rate=${LR} \
    --opt_lamb_beta_1=${OPT_LAMB_BETA_1} \
    --opt_lamb_beta_2=${OPT_LAMB_BETA_2} \
    --warmup_proportion=${WARMUP_PROPORTION} \
    --warmup_steps=${WARMUP_STEPS} \
    --start_warmup_step=${START_WARMUP_STEP} \
    --max_steps=${MAX_STEPS} \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=/workspace/data_phase2 \
    --init_checkpoint=/workspace/phase1/model.ckpt-28252.pt \
    "
PHASES=( "$PHASE1" "$PHASE2" )

cluster=''
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
    cluster='circe'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
    cluster='selene'
fi

declare -a CMD
if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( './bind.sh' '--cpu=exclusive' '--ib=single' '--cluster=${cluster}' '--' ${NSYSCMD} 'python' '-u')
else
    # docker or single gpu, no need to bind
    CMD=( ${NSYSCMD} 'python' '-u' )
fi

# Run fixed number of training samples
BERT_CMD="\
    ${CMD[@]} \
    /workspace/bert/run_pretraining.py \
    $PHASE2 \
    --do_train \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=${TARGET_MLM_ACCURACY} \
    --weight_decay_rate=${WEIGHT_DECAY_RATE} \
    --max_samples_termination=${MAX_SAMPLES_TERMINATION} \
    --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES} --eval_iter_samples=${EVAL_ITER_SAMPLES} \
    --eval_batch_size=16 --eval_dir=/workspace/evaldata --num_eval_examples 10000 \
    --cache_eval_data \
    --output_dir=/results \
    --fp16  \
    --distributed_lamb --dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-ag-pg=1 --dwu-num-blocks=1 \
    --gradient_accumulation_steps=${GRADIENT_STEPS} \
    --log_freq=0 \
    --bert_config_path=/workspace/phase1/bert_config.json "

if [ -n "${SLURM_LOCALID-}" ]; then
  BERT_CMD="${BERT_CMD} --local_rank=${SLURM_LOCALID} "
fi

if [[ $USE_DDP != 1 || $GRADIENT_STEPS != 1 ]]; then
    BERT_CMD="${BERT_CMD} --allreduce_post_accumulation --allreduce_post_accumulation_fp16"
fi

if [[ ${SYNTH_DATA} -ge 1 ]]; then
 EXTRA_PARAMS+=" --synthetic_input "
fi

# put this at the very end in case someone would explicitly override above defaults
BERT_CMD="${BERT_CMD}  ${EXTRA_PARAMS} "

if [[ $UNITTEST != 0 ]]; then
  BERT_CMD="NVIDIA_TF32_OVERRIDE=0 python /workspace/bert/unit_test/test_main.py"
fi

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly NODE_RANK="${SLURM_NODEID}"
  if [ "$NODE_RANK" -eq 0 ] && [ "$LOCAL_RANK" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

# Options

[ "${DEBUG}" = "1" ] && set -x

eval "${LOGGER} ${BERT_CMD} --seed=${SEED}"

set +x

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="bert"
echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"

