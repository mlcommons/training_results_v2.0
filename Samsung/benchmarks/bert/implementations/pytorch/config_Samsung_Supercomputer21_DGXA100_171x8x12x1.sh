## DL params
export BATCHSIZE=12
#export BATCHSIZE=32
export LR=0.00039299421615882676
export OPT_LAMB_BETA_1=0.8003575444254268
export OPT_LAMB_BETA_2=0.8174653681457493
export WARMUP_STEPS=82
export END_LR=0.0000494549100544884
export WEIGHT_DECAY_RATE=0.0028861703130042965
export EPSILON=1e-9
export START_STEPS=-16
export MAX_SAMPLES_TERMINATION=5000000
export MAX_STEPS=299
export EVAL_START=325000
export EVAL_ITER=325000
export GRAD_STEPS=1
export TARGET_MLM_ACCURACY=0.72
export OPTIMIZER="FusedAdam"

export EXTRA_PARAMS="--unpad --unpad_fmha --use_split_data --split_batch_cnt 4 2 2 4 --group_exchange_padding --ngpus_per_group=8  --reverse_indices --local_gradient_clip --use_partial_data --log_freq 0 --lr_max_steps 179 --fused_bias_fc --fused_bias_mha --fused_dropout_add"

export INPUT_DIR='/local_scratch/ci2015.choi/bert_v1.0/partial_data/4bin_1368gpu'
export EVAL_DIR='/local_scratch/ci2015.choi/bert_v1.0/eval_10k_tf'
export CHECKPOINTDIR_PHASE1='/local_scratch/ci2015.choi/bert_v1.0/checkpointdir_phase1'

## System run parms
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
