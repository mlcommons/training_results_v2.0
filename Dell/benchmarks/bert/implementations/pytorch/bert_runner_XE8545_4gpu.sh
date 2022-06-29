#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NEXP=10
source config_XE8545_1x4x56x1.sh

#nvcr.io/nvdlfwea/mlperfv11/bert:20211013.pytorch
CONT=6d4a2f12b93a RESULTSDIR=/root/mlperf_training/bert DATADIR=/mount/training_datasets_v2.0/bert/hdf5/training-4320/hdf5_4320_shards_varlength DATADIR_PHASE2=/mount/training_datasets_v2.0/bert/hdf5/training-4320/hdf5_4320_shards_varlength EVALDIR=/mount/training_datasets_v2.0/bert/hdf5/eval_varlength CHECKPOINTDIR=/root/mlperf_training/bert CHECKPOINTDIR_PHASE1=/mount/training_datasets_v2.0/bert/phase1 UNITTESTDIR=/xe8545_nvme0/scripts/bert/unit_test ./run_with_docker.sh
