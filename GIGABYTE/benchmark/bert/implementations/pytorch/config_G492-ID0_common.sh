## System config params
export DGXNGPU=8
export DGXSOCKETCORES=40
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/dataset/datasets/bert_new/hdf5/training-4320/hdf5_4320_shards_varlength" #"/raid/datasets/bert/hdf5/4320_shards"
export EVALDIR="/dataset/datasets/bert_new/hdf5/eval_varlength" #"/raid/datasets/bert/hdf5/eval_4320_shard"
export DATADIR_PHASE2="/dataset/datasets/bert_new/hdf5/training-4320/hdf5_4320_shards_varlength" #"/raid/datasets/bert/hdf5/4320_shards"
export CHECKPOINTDIR="/dataset/datasets/bert_new/phase1" #"./ci_checkpoints"
export RESULTSDIR="/dataset/training_v2.0/bert.20220509/logs" #"./results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/dataset/datasets/bert_new/phase1"  #"/raid/datasets/bert/checkpoints/checkpoint_phase1"
export UNITTESTDIR="$(pwd)/unit_test"  #"/lustre/fsw/mlperf/mlperft-bert/unit_test"
