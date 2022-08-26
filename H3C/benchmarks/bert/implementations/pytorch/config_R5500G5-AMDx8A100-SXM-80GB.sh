## DL params
export BATCHSIZE=56
export GRADIENT_STEPS=1
export LR=4e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=7280
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_R5500G5-AMDx8A100-SXM-80GB_common.sh
export NEXP=10
