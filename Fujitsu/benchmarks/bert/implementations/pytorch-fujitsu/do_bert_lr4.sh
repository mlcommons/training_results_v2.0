#! /bin/bash

bash ../send_msg_to_mattermost.sh "notsu $0 start"
source ./config_GX2570M6_lr4.sh

CONT=nvcr.io/nvdlfwea/mlperfv20/bert:20220509.pytorch
RESULTSDIR=$(realpath ../logs-bert-lr4)
CONT=$CONT LOGDIR=$RESULTSDIR DATADIR=$DATADIR DATADIR_PHASE2=$DATADIR_PHASE2 \
  EVALDIR=$EVALDIR CHECKPOINTDIR=$CHECKPOINTDIR CHECKPOINTDIR_PHASE1=$CHECKPOINTDIR_PHASE1 \
  NEXP=3 \
  ./run_with_docker.sh
bash ../send_msg_to_mattermost.sh "@notsu $0 end"
