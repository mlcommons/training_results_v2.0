#!/bin/bash

set -e
set -x

run_benchmark resnet v4-8192 "python3 resnet/resnet_main.py \
  --master=${MLP_TPU_NAME} \
  --data_dir=${MLP_RESNET_DATA}/imagenet-1024 \
  --eval_batch_size=32768 \
  --iterations_per_loop=157 \
  --enable_profiling=false \
  --resnet_depth=50 \
  --steps_per_eval=157 \
  --train_batch_size=32768 \
  --replicas_per_host=4 \
  --num_replicas=4096 \
  --logical_devices=1 \
  --bfloat16_replica_threshold=8192 \
  --dataset_threadpool_size=100 \
  --distributed_group_size=8 \
  --enable_lars=true \
  --weight_decay=.0001 \
  --label_smoothing=0.1 \
  --lars_base_learning_rate=27.50 \
  --lars_warmup_epochs=16 \
  --lars_epsilon=0.0 \
  --momentum=0.9321 \
  --use_space_to_depth=true \
  --tpu_topology_dim_count=3 \
  --sleep_after_init=360 \
  --train_steps=2355"
