## DL params
export BATCHSIZE=56
export GRADIENT_STEPS=1
#export LR=3.5e-4
export LR=4.0e-4
export MAX_SAMPLES_TERMINATION=4500000
#export MAX_STEPS=7100
export MAX_STEPS=7280
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

## Data Paths
export DATADIR="/mnt/data4/work/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/mnt/data4/work/bert_data/hdf5/eval_varlength"
export DATADIR_PHASE2="/mnt/data4/work/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export CHECKPOINTDIR="./ci_checkpoints"
export RESULTSDIR="./results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/mnt/data4/work/bert_data/phase1"
export UNITTESTDIR=$(realpath ./unit_test)
