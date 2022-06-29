source $(dirname ${BASH_SOURCE[0]})/config_XE8545x4A100-40GB_common.sh
export MXNET_DETERMINISTIC=1

## DL params
export OPTIMIZER="sgdwfastlars"

#45th 3.38
#export BATCHSIZE="102"
#export LR="20"
#export WD="2.5e-5"
#export EVAL_OFFSET="0" # Targeting epoch no. 45

#3.26
export BATCHSIZE="128" #43th 5.59, cannot coverged at 42
#export LR="17"
export LR="22"
export WD="2e-4"
export EVAL_OFFSET="2" # Targeting epoch no. 43


export KVSTORE="horovod"
export WARMUP_EPOCHS="9"
export EVAL_PERIOD="4"
export MOM="0.9"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS=${NUMEPOCHS:-"48"}

#export NETWORK="resnet-v1b-stats-fl"
export NETWORK="resnet-v1b-fl"

export DALI_THREADS="6"
export DALI_HW_DECODER_LOAD="0.7"
export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="12288"

#DALI buffer presizing hints
export DALI_PREALLOCATE_WIDTH="5980"
export DALI_PREALLOCATE_HEIGHT="6430"
export DALI_DECODER_BUFFER_HINT="1315942" #1196311*1.1
export DALI_CROP_BUFFER_HINT="165581" #150528*1.1
export DALI_TMP_BUFFER_HINT="355568328" #871491*batch_size
export DALI_NORMALIZE_BUFFER_HINT="441549" #401408*1.1

export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999

## System run parms
export DGXNNODES=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00

#export NCCL_SOCKET_IFNAME=
export NCCL_MAX_RINGS=4
export OMPI_MCA_btl_openib_if_include=mlx5_0:1
export UCX_NET_DEVICES=mlx5_0:1

