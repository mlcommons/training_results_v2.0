#!/bin/bash

set -e
set -x

run_benchmark bert v4-6912 "python3 bert/run_pretraining.py \
  --master=${MLP_TPU_NAME} \
  --num_warmup_steps=0 \
  --learning_rate=0.0029293 \
  --weight_decay_rate=0.001  \
  --beta_1=0.7206 \
  --beta_2=0.78921 \
  --log_epsilon=-6 \
  --num_eval_samples=10000 \
  --stop_steps=700 \
  --repeatable=false \
  --stop_threshold=0.72 \
  --input_file=${MLP_BERT_DATA}/seq_512_mpps_76_tfrecords4/part-* \
  --eval_input_file=${MLP_BERT_DATA}/eval_10k/* \
  --init_checkpoint=${MLP_BERT_DATA}/converted \
  --bert_config_file=${MLP_BERT_DATA}/bert_config.json \
  --train_batch_size=6912 \
  --eval_batch_size=6912 \
  --max_eval_steps=2 \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --num_train_steps=700 \
  --start_warmup_step=-700 \
  --save_checkpoints_steps=33 \
  --iterations_per_loop=33 \
  --optimizer=lamb \
  --use_bfloat16_activation=true \
  --use_bfloat16_all_reduce=true \
  --do_train=true \
  --do_eval=false \
  --use_tpu=true \
  --enable_profiling=false \
  --num_tpu_cores=3456 \
  --replicas_per_host=4 \
  --sleep_after_init=650 \
  --steps_per_update=1 \
  --clip_by_global_norm_after_grad=false"
