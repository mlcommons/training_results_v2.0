## DL params
export BATCHSIZE=2
export GRADIENT_STEPS=1
export PACKING_FACTOR=2
export INIT_LOSS_SCALE=128.0
export LR=0.0033
export MAX_SAMPLES_TERMINATION=12000000
export MAX_STEPS=470
export OPT_LAMB_BETA_1=0.75
export OPT_LAMB_BETA_2=0.9
export START_WARMUP_STEP=-100
export WEIGHT_DECAY_RATE=0.0166629
export WARMUP_STEPS=290
export SBATCH_NETWORK=sharp
export NCCL_GRAPH_REGISTER=1
export EXTRA_PARAMS="--use_cuda_graph --pad_fmha --cuda_graph_mode 'full_iteration' --max_iterations_per_graph 1 --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_bias_fc_loss_head --packed_samples "
export PHASE=2

## System run parms
export DGXNNODES=512

# hparams that depend on number of nodes
export EVAL_ITER_START_SAMPLES=325000 #$(echo "25000*(0.05*(230.23*${BATCHSIZE}*${DGXNNODES}*8*${PACKING_FACTOR}+3000000)/25000)" | bc)
export EVAL_ITER_SAMPLES=${EVAL_ITER_START_SAMPLES}

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=7

export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
export DATADIR_PHASE2="/raid/datasets/bert/hdf5/4320_packed_shards"

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1
