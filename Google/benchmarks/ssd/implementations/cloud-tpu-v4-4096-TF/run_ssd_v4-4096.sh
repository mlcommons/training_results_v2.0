#!/bin/bash
set -e
set -x

run_benchmark ssd v4-4096 "python3 ssd/ssd_main.py \
  --bfloat16_replica_threshold=8192 \
  --dataset_threadpool_size=160 \
  --enable_profiling=False \
  --eval_batch_size=28672 \
  --hparams=use_bfloat16=true,conv0_space_to_depth=true,distributed_group_size=64,base_learning_rate=1e-4,lr_warmup_epoch=1,weight_decay=0,logical_devices=1,tpu_topology_dim_count=3 \
  --iterations_per_loop=571 \
  --master=${MLP_TPU_NAME} \
  --num_epochs=25 \
  --num_shards=2048 \
  --eval_samples=24781 \
  --num_examples_per_epoch=1170301 \
  --replicas_per_host=4 \
  --resnet_checkpoint=${MLP_DATA_PATH}/openimages-mlperf_2.0/ssd_checkpoint_mlperf_2.0/retinanet_restnext50.ckpt \
  --sleep_after_init=1300 \
  --train_batch_size=2048 \
  --training_file_pattern=${MLP_SSD_DATA}/train* \
  --val_json_file=${$MLP_DATA_PATH}/openimages-mlperf_2.0/openimages-mlperf.json \
  --validation_file_pattern=${MLP_SSD_DATA}/val*"
