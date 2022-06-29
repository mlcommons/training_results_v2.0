#!/bin/bash

function cleanup() {
  echo "Cleanup"
  # TBD
}

function submission_run() {
  MODEL=$1
  NUM_RUNS=$2
  TOPOLOGY=$3

  echo "Running $MODEL on $TOPOLOGY for $NUM_RUNS runs."
  export "CLOUDSDK_PYTHON=/usr/bin/python3"

  BENCHMARK=$MODEL-$TOPOLOGY
  MLP_SUBMISSION_DIR=$MLP_GCS_SUBMISSION_RESULTS/$BENCHMARK/$(date +%s)
  RESULTS_DIR=$LOCAL_RESULTS_DIR/$BENCHMARK
  mkdir -p $RESULTS_DIR

  CMD="$4"

  cleanup

  export TPU_LOAD_LIBRARY=0
  cd $TOP_LEVEL_DIR

  for run_index in $(seq 1 $NUM_RUNS); do
    echo "Starting run $run_index"
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)

    echo "STARTING TIMING RUN AT $start_fmt"
    # Commented out because for the submission package we do not access git
    # GET_GIT_HASH=$(git rev-parse HEAD)
    # echo "Built from commit hash $GET_GIT_HASH" > $RESULTS_DIR/results.txt
    echo "Cloud run command: $CMD" >> $RESULTS_DIR/results.txt
    eval $CMD 2>&1 | tee -a $RESULTS_DIR/results.txt

    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"

    gsutil -m cp $RESULTS_DIR/results.txt $MLP_SUBMISSION_DIR/$MODEL_$TOPOLOGY_log_$run_index

    # report result 
    result=$(( $end - $start )) 
    echo "RESULT,$MODEL,0,$result,$USER,$start_fmt"
  done

  cd -
}

function upload_results() {
  RESULTS_DIR=$1
  GCS_DEST=$2

  echo "Uploading logs"
  gsutil -m cp $RESULTS_DIR/results.txt ${MLP_GCS_MODEL_DIR}/
}

function run_benchmark() {
  MODEL=$1
  TOPOLOGY=$2
  export "CLOUDSDK_PYTHON=/usr/bin/python3"

  BENCHMARK=$MODEL-$TOPOLOGY
  MLP_GCS_MODEL_DIR=$MLP_GCS_RUNS_RESULTS/$BENCHMARK/$(date +%s)
  RESULTS_DIR=$LOCAL_RESULTS_DIR/$BENCHMARK
  mkdir -p $RESULTS_DIR

  CMD="$3 --model_dir=$MLP_GCS_MODEL_DIR"

  cleanup

  export TPU_LOAD_LIBRARY=0
  cd $TOP_LEVEL_DIR
  start=$(date +%s)
  start_fmt=$(date +%Y-%m-%d\ %r)

  echo "STARTING TIMING RUN AT $start_fmt"

  # Commented out because for the submission package we do not access git
  # GET_GIT_HASH=$(git rev-parse HEAD)
  # echo "Built from commit hash $GET_GIT_HASH" > $RESULTS_DIR/results.txt
  echo "Cloud run command: $CMD" >> $RESULTS_DIR/results.txt
  eval $CMD 2>&1 | tee -a $RESULTS_DIR/results.txt

  # end timing
  end=$(date +%s)
  end_fmt=$(date +%Y-%m-%d\ %r)
  echo "ENDING TIMING RUN AT $end_fmt"

  upload_results $RESULTS_DIR $MLP_GCS_MODEL_DIR

  # report result 
  result=$(( $end - $start )) 
  echo "RESULT,$MODEL,0,$result,$USER,$start_fmt"

  cd -
}

export -f cleanup
export -f upload_results
export -f run_benchmark
export -f submission_run

export MLP_TPU_NAME=$TPU_NAME
export MLP_PROJECT_NAME=$PROJECT_NAME
export MLP_BUCKET_PATH=$BUCKET
export MLP_DATA_PATH=$MLP_BUCKET_PATH/data

if [[ -z "${LOCAL_DATA_PATH}" ]]; then
  export MLP_RESNET_DATA=$MLP_DATA_PATH
  export MLP_MASKRCNN_DATA=$MLP_DATA_PATH/maskrcnn-coco
  export MLP_SSD_DATA=$MLP_DATA_PATH/openimages-mlperf_2.0
  export MLP_DLRM_DATA=$MLP_DATA_PATH/criteo-dlrm
  export MLP_UNET_DATA=$MLP_DATA_PATH/kits19
  export MLP_BERT_DATA=$MLP_DATA_PATH/bert_pretrain
else
  export MLP_RESNET_DATA=$LOCAL_DATA_PATH
  export MLP_MASKRCNN_DATA=$LOCAL_DATA_PATH/maskrcnn-coco
  export MLP_SSD_DATA=$LOCAL_DATA_PATH/openimages-mlperf_2.0_1024_shard
  export MLP_DLRM_DATA=$LOCAL_DATA_PATH
  export MLP_UNET_DATA=$LOCAL_DATA_PATH/kits19
  export MLP_BERT_DATA=$LOCAL_DATA_PATH/bert_pretrain
fi

export MLP_GCS_RUNS_RESULTS=$MLP_BUCKET_PATH/runs
export MLP_GCS_SUBMISSION_RESULTS=$MLP_BUCKET_PATH/submission_runs

export MLP_ZONE=us-central1-b

PYTHONPATH=""
export PYTHONPATH=`pwd`/$TOP_LEVEL_DIR:$PYTHONPATH

export LOCAL_RESULTS_DIR=~/results
mkdir -p $LOCAL_RESULTS_DIR

