## DL params
export BATCHSIZE=56
export GRADIENT_STEPS=2
export LR=3.5e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=7100
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_stream_attn --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
# export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
# https://stackoverflow.com/questions/9901210/bash-source0-equivalent-in-zsh
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]:-${(%):-%x}}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
# source ${BASH_SOURCE%/*}/config_DGXA100_common.sh
# source ${BASH_SOURCE[0]:-${(%):-%x}%/*}/config_DGXA100_common.sh
source ${PWD}/config_DGXA100_common.sh
