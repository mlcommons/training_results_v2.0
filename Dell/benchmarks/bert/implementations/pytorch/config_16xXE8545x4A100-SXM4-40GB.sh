## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=1.5e-3
#export LR=0.002
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=1271
export OPT_LAMB_BETA_1=0.83
export OPT_LAMB_BETA_2=0.925
export START_WARMUP_STEP=-25
export WARMUP_STEPS=100

export SBATCH_NETWORK=sharp
#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gemm_gelu "
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gemm_gelu "
#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=4 "
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
#export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=175000
#export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=15
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_XE8545x4A100-SXM-40GB_common.sh
NCCL_SOCKET_IFNAME=mlx5_0,mlx5_1
