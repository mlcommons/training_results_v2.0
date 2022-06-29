## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3"

## Data Paths
export DATADIR="/raid/datasets/bert/hdf5/4320_shards"
export EVALDIR="/raid/datasets/bert/hdf5/eval_4320_shard"
export DATADIR_PHASE2="/raid/datasets/bert/hdf5/4320_shards"
export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
export RESULTSDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/raid/datasets/bert/checkpoints/checkpoint_phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
