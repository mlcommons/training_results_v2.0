#!/bin/bash
# Note: need to specify hosts and DGXSYSTEM
export hosts=('192.168.16.1' '192.168.16.2')
source config_${DGXSYSTEM}.sh
export PORT=4332
export NGPU=8

echo "Total Node : ${#hosts[*]}"
export NCORES_PER_SOCKET=$(ssh ${hosts[0]} "/usr/bin/lscpu | grep 'Core(s)' | awk {'print \$4'}")
SET=$(seq 0 $[${#hosts[*]}-1])
SSH='ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no'
BATCHSIZE=${BATCHSIZE:-3}
LR=${LR:-0.00035}
WARMUP_PORTION=${WARMUP_PORTION:-0.0}
END_LR=${END_LR:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-0}
# MLPerf BERT v1.0, data splitting 4-bin partial data
INPUT_DIR=${INPUT_DIR:-'/data/bert_v1.0/partial_data/4bin'}
EVAL_DIR=${EVAL_DIR:-'/data/bert_v1.0/eval_10k_tf'}
MAX_SAMPLES_TERMINATION=${MAX_SAMPLES_TERMINATION:-4500000}
EVAL_START=${EVAL_START:-0}
EVAL_ITER=${EVAL_ITER:-100000}
START_STEPS=${START_STEPS:=0}
MAX_STEPS=${MAX_STEPS:-13700}
GRAD_STEPS=${GRAD_STEPS:-1}
EPSILON=${EPSILON:-1e-6}
OPT_LAMB_BETA_1=${OPT_LAMB_BETA_1:-0.9}
OPT_LAMB_BETA_2=${OPT_LAMB_BETA_2:-0.999}
OPTIMIZER=${OPTIMIZER:-'FusedAdam'}
TARGET_MLM_ACCURACY=${TARGET_MLM_ACCURACY:-0.8}
CLIP_VALUE=${CLIP_VALUE:-0.0}
WEIGHT_DECAY_RATE=${WEIGHT_DECAY_RATE:-0.01}
EXTRA_PARAMS=${EXTRA_PARAMS:-""}
LOGDIR=${LOGDIR:-'./results'}
#MLPef BERT v1.0 checkpoint
CHECKPOINTDIR_PHASE1=${CHECKPOINTDIR_PHASE1:-'/data/bert_v1.0/checkpointdir_phase1'}
GBS=$(( 8*$BATCHSIZE*${#hosts[*]} ))
now=$(date +"%m%d_%I_%M")
CLEAR_CACHE="from mlperf_logging.mllog import constants; from mlperf_logger import log_event; log_event(key=constants.CACHE_CLEAR, value=True)"
EXP=0
NEXP=${NEXP:-1}
for nexp in `seq 0 $[NEXP-1]`
do
	#seed=${nexp}
	SEED=${RANDOM}
	echo "Start $nexp expirement"
	(
	for h in $SET
	do
		hostn="${hosts[$h]}"
		#$(eval echo $SSH) $hostn "sync && sudo /sbin/sysctl vm.drop_caches=3; cd ${PWD}; ${PYTHON} -u -c  \"from mlperf_logging.mllog import constants; from mlperf_logger import log_event; log_event(key=constants.CACHE_CLEAR, value=True)\" "
		$(eval echo $SSH) $hostn "cd ${PWD}; ${PYTHON} -u -c  \"from mlperf_logging.mllog import constants; from mlperf_logger import log_event; log_event(key=constants.CACHE_CLEAR, value=True)\" "

		$(eval echo $SSH) $hostn "cd $PWD; NCCL_IB_HCA=mlx5_1,mlx5_0,mlx5_5,mlx5_2 $PYTHON -u -m bind_launch --nnodes=${#hosts[*]} --node_rank=$h \
		--master_addr=${hosts[0]} --master_port=$PORT --nsockets_per_node=2 \
		--ncores_per_socket=$NCORES_PER_SOCKET --nproc_per_node=$NGPU \
		run_pretraining.py \
		--train_batch_size=$BATCHSIZE \
		--learning_rate=$LR \
		--epsilon=$EPSILON \
		--weight_decay_rate=$WEIGHT_DECAY_RATE \
		--end_learning_rate=$END_LR \
		--opt_lamb_beta_1=$OPT_LAMB_BETA_1 \
		--opt_lamb_beta_2=$OPT_LAMB_BETA_2 \
		--warmup_proportion=$WARMUP_PORTION \
		--start_warmup_step=$START_STEPS \
		--warmup_steps=$WARMUP_STEPS \
		--max_steps=$MAX_STEPS \
		--phase2 --max_seq_length=512 --max_predictions_per_seq=76 \
		--input_dir=$INPUT_DIR \
		--init_checkpoint=$CHECKPOINTDIR_PHASE1/model.ckpt-28252.pt --do_train --skip_checkpoint \
		--train_mlm_accuracy_window_size=0 \
		--target_mlm_accuracy=$TARGET_MLM_ACCURACY --max_samples_termination=$MAX_SAMPLES_TERMINATION \
		--eval_iter_start_samples=$EVAL_START --eval_iter_samples=$EVAL_ITER \
		--eval_batch_size=16 \
		--eval_dir=$EVAL_DIR \
		--output_dir=./results --fp16 --fused_gelu_bias --fused_mha --dense_seq_output --allreduce_post_accumulation --log_freq=1 \
		--allreduce_post_accumulation_fp16 \
		--gradient_accumulation_steps=$GRAD_STEPS \
		--optimizer=$OPTIMIZER \
		--bert_config_path=$CHECKPOINTDIR_PHASE1/bert_config.json --output_dir=./results --use_env \
		--use_apex_amp --cache_eval_data --seed ${seed} $EXTRA_PARAMS " &
		pids+=($!);
	done
	wait "${pids[@]}"
	) |& tee ${LOGDIR}/GBS${GBS}_${OPTIMIZER}_${now}_EXP${EXP}.log
sleep 5s
EXP=$(($EXP+1))
done
