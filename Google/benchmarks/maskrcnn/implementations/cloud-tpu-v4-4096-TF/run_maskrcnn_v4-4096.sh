#!/bin/bash

set -e
set -x

run_benchmark maskrcnn v4-4096 "numactl --cpunodebind=1 python3 mask_rcnn/mask_rcnn_main.py \
  --master=${MLP_TPU_NAME} \
  --resnet_checkpoint=${MLP_MASKRCNN_DATA}/model.ckpt-100080 \
  --hparams=first_lr_drop_step=6000,second_lr_drop_step=8000,lr_warmup_step=1905,learning_rate=0.26,shuffle_buffer_size=2048 \
  --num_shards=256 \
  --replicas_per_host=1 \
  --training_file_pattern=${MLP_MASKRCNN_DATA}/train* \
  --input_partition_dims=1 \
  --input_partition_dims=8 \
  --input_partition_dims=1 \
  --train_batch_size=256 \
  --eval_batch_size=512 \
  --num_all_tpu_cores=4096 \
  --sleep_after_init=600 \
  --validation_file_pattern=${MLP_MASKRCNN_DATA}/val* \
  --val_json_file=${MLP_DATA_PATH}/maskrcnn-coco/instances_val2017.json \
  --enable_profiling=False \
  --num_epochs=23"
  
