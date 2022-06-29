#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time_multi.sh
#set -x

#echo SLURM_LOCALID1=${SLURM_LOCALID}
source ./config_${DGXSYSTEM}.sh

#echo SLURM_LOCALID2=${SLURM_LOCALID}

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
#: "${OMPI_COMM_WORLD_LOCAL_RANK:=""}"
: "${SEED:=$RANDOM}"
: "${SLURM_JOB_ID:=$RANDOM}"
#: "${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK}}"
#SLURM_LOCALID=${OMPI_COMM_WORLD_LOCAL_RANK}
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
#echo "$TIME_TAGS"
#echo "$NVTX_FLAG"
#echo "$NCCL_TEST"
#echo "$EPOCH_PROF"
#echo "$SYNTH_DATA"

#echo SLURM_LOCALID3=${SLURM_LOCALID}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

if [ ! -z "$THROUGHPUT_RUN" ]
then
  MAX_STEPS=4
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
#echo SLURM_LOCALID4=${SLURM_LOCALID}

cluster=''
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
    cluster='circe'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
    cluster='selene'
fi
if [[ "${DGXSYSTEM}" == *XE8545* ]]; then
    cluster='rattler2'
fi

##source ./config_${DGXSYSTEM}.sh
#echo SLURM_LOCALID=${SLURM_LOCALID}
#echo SLURM_NTASKS=$SLURM_NTASKS
#echo SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES
##export RANK=$OMPI_COMM_WORLD_NODE_RANK
export RANK=$SLURM_PROCID
#echo RANK=$SLURM_PROCID
##export RANK=2
##export WORLD_SIZE=8
export WORLD_SIZE=${SLURM_NTASKS}
echo WORLD_SIZE=$WORLD_SIZE
##export MASTER_ADDR="node020"
export MASTER_ADDR=$( echo $SLURM_JOB_NODELIST | cut -d '-' -f1 | cut -d '-' -f2 - | tr -d '[' )
echo MASTER_ADDR=$MASTER_ADDR
##export MASTER_ADDR=10.11.0.20
export MASTER_PORT=19002
echo HOSTNAME=$HOSTNAME
#

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    #CMD=( './bind.sh' '--cpu=exclusive' '--ib=single' '--cluster=${cluster}' '--' 'python' '-u')
#    CMD=( './bind.sh' '--cpu=exclusive' '--cluster=${cluster}' '--' 'python' '-u')
    #CMD=( './bind.sh' '--cpu=exclusive' '--cluster=rattler2' '--' 'python' '-u')
#    CMD=( './bind.sh' '--cpu=exclusive,nosmt' '--cluster=rattler2' '--' 'python' '-u')
#     CMD=( '/bm_utils/bind.sh' '--cpu=/bm_utils/azure.sh' '--mem=/bm_utils/azure.sh' '--ib=single' "--cluster=${cluster}" '--' 'python' '-u')

     CMD=( './bind.sh' '--cpu=xe8545_topology.sh' '--mem=xe8545_topology.sh' '--ib=single' '--cluster=${cluster}' '--' 'python' '-u' )
    #CMD=( 'python' '-u' '-m' 'torch.distributed.run' '--nproc_per_node=4')
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( 'python' '-u' '-m' 'bind_pyt' "--nsockets_per_node=${DGXNSOCKET}" \
    "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
fi
     #CMD=( './bind.sh' '--cpu=xe8545_topology.sh' '--mem=xe8545_topology.sh' '--cluster=rattler2' '--' ${NSYSCMD} 'python' '-u' )
#    CMD=( 'python' '-u' )

# Run fixed number of training samples
BERT_CMD="\
    ${CMD[@]} \
    ${NSYSCMD} \
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
    --fp16 \
    --distributed_lamb --dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-ag-pg=1 --dwu-num-blocks=1 \
    --gradient_accumulation_steps=${GRADIENT_STEPS} \
    --log_freq=0 \
    --bert_config_path=/workspace/phase1/bert_config.json "
#v1.1 used this
#    --fp16 --fused_bias_fc --fused_bias_mha --fused_dropout_add \

#echo BERT_CMD_SLURM_LOCALID_local_rank3=${SLURM_LOCALID-}
if [ -n "${SLURM_LOCALID-}" ]; then
  BERT_CMD="${BERT_CMD} --local_rank=${SLURM_LOCALID} "
fi

if [[ $USE_DDP != 1 || $GRADIENT_STEPS != 1 ]]; then
    BERT_CMD="${BERT_CMD} --allreduce_post_accumulation --allreduce_post_accumulation_fp16"
fi

# put this at the very end in case someone would explicitly override above defaults
BERT_CMD="${BERT_CMD}  ${EXTRA_PARAMS} "
#echo EXTRA_PARAMS=${EXTRA_PARAMS}

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

