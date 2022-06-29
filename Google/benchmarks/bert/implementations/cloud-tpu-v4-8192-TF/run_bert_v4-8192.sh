#!/bin/bash

set -e
set -x

run_benchmark bert_bucketized v4-8192 "python3 bert/run_pretraining.py \
  --master=${MLP_TPU_NAME} \
  --num_warmup_steps=314 \
  --learning_rate=0.0034064947274845907 \
  --weight_decay_rate=0.021344844194329957 \
  --beta_1=0.8069036244593849 \
  --enable_profiling=false \
  --beta_2=0.8731040542461879 \
  --log_epsilon=-6 \
  --num_eval_samples=10000 \
  --stop_steps=700 \
  --repeatable=false \
  --stop_threshold=0.72 \
  --input_file=${MLP_BERT_DATA}/seq_512_mpps_76_tfrecords4_1k/part-* \
  --eval_input_file=${MLP_BERT_DATA}/eval_10k/* \
  --init_checkpoint=${MLP_BERT_DATA}/converted \
  --bert_config_file=${MLP_BERT_DATA}/bert_config.json \
  --train_batch_size=14336 \
  --eval_batch_size=12288 \
  --max_eval_steps=1 \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --num_train_steps=454 \
  --start_warmup_step=-112 \
  --save_checkpoints_steps=33 \
  --iterations_per_loop=21 \
  --optimizer=lamb \
  --use_bfloat16_activation=true \
  --use_bfloat16_all_reduce=true \
  --do_train=true \
  --do_eval=false \
  --use_tpu=true \
  --num_tpu_cores=4096 \
  --replicas_per_host=4 \
  --sleep_after_init=1130 \
  --steps_per_update=1 \
  --seq_len_buckets=64 \
  --seq_len_buckets=128 \
  --seq_len_buckets=192 \
  --seq_len_buckets=256 \
  --seq_len_buckets=304 \
  --seq_len_buckets=512 \
  --batch_size_buckets=14 \
  --batch_size_buckets=8 \
  --batch_size_buckets=5 \
  --batch_size_buckets=4 \
  --batch_size_buckets=3 \
  --batch_size_buckets=2 \
  --clip_by_global_norm_after_grad=false"
