## DL params
export BATCHSIZE=2
export GRADIENT_STEPS=1
export LR=0.00255
export OPT_LAMB_BETA_1=0.71
export OPT_LAMB_BETA_2=0.88
export WEIGHT_DECAY_RATE=0.0
export MAX_SAMPLES_TERMINATION=7000000
export MAX_STEPS=820
export WARMUP_STEPS=256
export START_WARMUP_STEP=-76
export WEIGHT_DECAY_RATE=0.0166629
export SBATCH_NETWORK=sharp
export NCCL_GRAPH_REGISTER=1
#export EXTRA_PARAMS="--use_cuda_graph --pad_fmha --cuda_graph_mode 'full_iteration' --max_iterations_per_graph 1 --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_bias_fc_loss_head  --fused_gemm_gelu --packed_samples "
export EXTRA_PARAMS="--use_cuda_graph --pad_fmha --cuda_graph_mode 'full_iteration' --max_iterations_per_graph 1 --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_bias_fc_loss_head --packed_samples " # --dwu_num_blocks=4 --dwu_overlap_reductions "
export PHASE=2

export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

## System run parms
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_MINUTES=7
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
export DATADIR_PHASE2="/raid/datasets/bert/hdf5/4320_packed_shards"

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1
