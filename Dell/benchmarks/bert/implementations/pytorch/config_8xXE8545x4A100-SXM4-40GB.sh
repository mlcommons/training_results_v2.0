## DL params
#export BATCHSIZE=112
#export GRADIENT_STEPS=2
#export BATCHSIZE=64
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=0.002
#export LR=4e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=2254
export OPT_LAMB_BETA_1=0.66
export OPT_LAMB_BETA_2=0.996
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding"
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu "

export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
#source ${BASH_SOURCE%/*}/config_DGXA100_4gpu_common.sh
source config_XE8545x4A100-SXM-40GB_common.sh
NCCL_SOCKET_IFNAME=mlx5_0,mlx5_1
