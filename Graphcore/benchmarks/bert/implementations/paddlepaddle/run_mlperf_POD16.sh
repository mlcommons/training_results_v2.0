# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"./profiles/Paddle_POD16","autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true"}'

seed=${1}
submission_run_index=${2}

python3.7 bert.py \
        --POD 16 \
        --input_files /paddle_develop/jan2020_three_seq_per_pack_with_duplications \
        --output_dir pretrain_model \
        --seq_len 512 \
        --hidden_size 1024 \
        --vocab_size 30912 \
        --max_predictions_per_seq 76 \
        --max_position_embeddings 512 \
        --num_hidden_layers 24 \
        --max_training_sequences 3200000 \
        --learning_rate 3.5e-3 \
        --weight_decay 1e-3 \
        --max_steps 1000 \
        --warmup_steps 0 \
        --seed ${seed} \
        --beta1 0.55 \
        --beta2 0.885 \
        --device ipu \
        --micro_batch_size 3 \
        --scale_loss 10.0 \
        --optimizer_type lamb \
        --enable_pipelining True \
        --batches_per_step 1 \
        --enable_replica True \
        --local_num_replicas 2 \
        --enable_grad_acc True \
        --grad_acc_factor 256 \
        --batch_size 1536 \
        --no_attn_dropout False \
        --hidden_dropout_prob 0.1 \
        --attention_probs_dropout_prob 0.1 \
        --activation_checkpoint_dtype FLOAT8 \
        --duplication_factor 3 \
        --max_sequences_per_pack 3 \
        --shuffle True \
        --tf_checkpoint /paddle_develop/model.ckpt-28252 \
        --use_prepacked_pretraining_dataset True \
        --avg_seq_per_pack 2 \
        --accuracy_averaging_basis pertoken \
        --wandb False \
        --enable_validation True \
        --merge_collectives True \
        --split_qkv False \
        --optimizer_state_offchip False \
        --validation_files /paddle_develop/jan2020_packed_eval_10k \
        --submission_run_index ${submission_run_index}
