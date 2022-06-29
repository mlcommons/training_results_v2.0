## System config params
export DGXNGPU=8
# export DGXSOCKETCORES=64
# Gcloud seems to have 2 sockets, 24 CPUs per socket, 2 threads per CPU
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
# export DATADIR="/raid/datasets/bert/hdf5/4320_shards"
# export EVALDIR="/raid/datasets/bert/hdf5/eval_4320_shard"
# export DATADIR_PHASE2="/raid/datasets/bert/hdf5/4320_shards"
# export DATADIR="/home/user/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
# export EVALDIR="/home/user/bert_data/hdf5/eval_varlength"
# export DATADIR_PHASE2="/home/user/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR="/mnt/data/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/mnt/data/bert_data/hdf5/eval_varlength"
export DATADIR_PHASE2="/mnt/data/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
# export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
export CHECKPOINTDIR="/mnt/data/ci_checkpoints"
#using existing checkpoint_phase1 dir
# export CHECKPOINTDIR_PHASE1="/raid/datasets/bert/checkpoints/checkpoint_phase1"
export CHECKPOINTDIR_PHASE1="/mnt/data/bert_data/phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
