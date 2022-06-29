## System config params
export DGXNGPU=8
export DGXSOCKETCORES=40
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/raid0/datasets/bert_v2.0/hdf5"
export EVALDIR="/raid0/datasets/bert_v2.0/hdf5/eval_varlength"
export DATADIR_PHASE2="/raid0/datasets/bert_v2.0/hdf5/training-4320/hdf5_4320_shards_varlength"
export CHECKPOINTDIR="/raid0/datasets/bert_v2.0/phase1"
export RESULTSDIR="results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/raid0/datasets/bert_v2.0/phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
