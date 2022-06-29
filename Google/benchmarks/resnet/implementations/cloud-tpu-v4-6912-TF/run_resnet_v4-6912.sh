#!/bin/bash

set -e
set -x

run_benchmark resnet v4-6912 "python3 resnet/resnet_main.py \
  --master=${MLP_TPU_NAME} \
  --data_dir=${MLP_RESNET_DATA}/imagenet-tensorflow_3456 \
  --eval_batch_size=55296 \
  --iterations_per_loop=93 \
  --resnet_depth=50 \
  --enable_profiling=false \
  --steps_per_eval=93 \
  --train_batch_size=55296 \
  --replicas_per_host=4 \
  --num_replicas=3456 \
  --logical_devices=1 \
  --bfloat16_replica_threshold=8192 \
  --dataset_threadpool_size=100 \
  --distributed_group_size=4 \
  --sleep_after_init=420 \
  --enable_lars=true \
  --weight_decay=.0001 \
  --label_smoothing=0.1 \
  --lars_base_learning_rate=25.67 \
  --lars_warmup_epochs=29 \
  --lars_epsilon=0.0 \
  --momentum=0.94053 \
  --use_space_to_depth=true \
  --sleep_after_init=360 \
  --tpu_topology_dim_count=3 \
  --train_steps=1860"
