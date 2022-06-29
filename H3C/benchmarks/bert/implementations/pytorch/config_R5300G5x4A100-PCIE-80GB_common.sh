## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3"

## Data Paths
export DATADIR="/Data/training/training/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/Data/training/training/bert_data/hdf5/eval_varlength"
export DATADIR_PHASE2="/Data/training/training/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength"
export CHECKPOINTDIR="/Data/training/training/bert_data/phase1"
#export RESULTSDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/Data/training/training/bert_data/phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
