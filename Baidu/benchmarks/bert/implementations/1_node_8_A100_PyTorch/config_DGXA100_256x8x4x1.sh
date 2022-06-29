## DL params
# steps500_lr0.004_wup0.1_b1_0.878_b2_0.974
export BATCHSIZE=4
export GRADIENT_STEPS=1
#export INIT_LOSS_SCALE=16384
export LR=0.00288293
export MAX_SAMPLES_TERMINATION=7000000
export MAX_STEPS=${MAX_STEPS:-600}
export OPT_LAMB_BETA_1=0.88
export OPT_LAMB_BETA_2=0.88
export START_WARMUP_STEP=-76
export WEIGHT_DECAY_RATE=0.0166629
export WARMUP_STEPS=287
export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--use_cuda_graph --pad_fmha --cuda_graph_mode 'full_iteration' --max_iterations_per_graph 1  "
export PHASE=2
export EVAL_ITER_START_SAMPLES=225000
export EVAL_ITER_SAMPLES=225000

## System run parms
export DGXNNODES=256
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:70:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1
