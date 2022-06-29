#! /bin/bash


export CONT=gitlab-master.nvidia.com:5005/dl/mlperf/training/maskrcnn
#export CLEAR_CACHES=0 # otherwise need sudo...
source config_DGXA100_1GPU_BS1.sh

#dlcluster
export DATADIR='/mnt/nvdl/datasets/coco_master/coco/coco-2017/'
export LOGDIR='/mnt/nvdl/usr/yudong/gitlab/maskrcnn/logs'
export PKLDIR=$DATADIR/pkl_coco
export SRCDIR="/mnt/nvdl/usr/yudong/gitlab/mlperf/maskrcnn/optimized/object_detection/pytorch"

#computelab
#export DATADIR='/scratch/local/mlperft/mrcnn/'
#export DATADIR='/scratch/local/yudong/mrcnn/logs/'
#export PKLDIR=$DATADIR/pkl_coco

#selene
# TODO: add selene path

readonly _cont_name=object_detection
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${DATADIR}/coco2017:/coco" "--volume=${LOGDIR}:/results" "--volume=${PKLDIR}:/pkl_coco" "--volume=${SRCDIR}:/workspace/object_detection")

# start the docker interactively
nvidia-docker run --rm -it \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" /bin/bash


#Run the cmd below to test
#pip install -v -e maskrcnn/
#python -u maskrcnn/tools/train_mlperf.py --config-file maskrcnn/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml DTYPE float16 PATHS_CATALOG maskrcnn/maskrcnn_benchmark/config/paths_catalog_dbcluster.py MODEL.WEIGHT /coco/models/R-50.pkl DISABLE_REDUCED_LOGGING True SOLVER.BASE_LR 0.00125 SOLVER.WARMUP_FACTOR 0.0000025 SOLVER.WARMUP_ITERS 500 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS '(1152000,1536000)' SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000 MODEL.RPN.FPN_POST_NMS_TOP_N_PER_IMAGE True NHWC True SOLVER.MAX_ITER 20 DATALOADER.DALI False DATALOADER.DALI_ON_GPU False DATALOADER.CACHE_EVAL_IMAGES True EVAL_SEGM_NUMPROCS 10 USE_CUDA_GRAPH True EVAL_MASK_VIRTUAL_PASTE True MODEL.BACKBONE.INCLUDE_RPN_HEAD True DATALOADER.NUM_WORKERS 1 PRECOMPUTE_RPN_CONSTANT_TENSORS True DATALOADER.HYBRID True CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION 1
#nsys profile --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/test.qdrep python -u maskrcnn/tools/train_mlperf.py --config-file maskrcnn/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml DTYPE float16 PATHS_CATALOG maskrcnn/maskrcnn_benchmark/config/paths_catalog_dbcluster.py MODEL.WEIGHT /coco/models/R-50.pkl DISABLE_REDUCED_LOGGING True SOLVER.BASE_LR 0.00125 SOLVER.WARMUP_FACTOR 0.0000025 SOLVER.WARMUP_ITERS 500 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS '(1152000,1536000)' SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000 MODEL.RPN.FPN_POST_NMS_TOP_N_PER_IMAGE True NHWC True SOLVER.MAX_ITER 20 DATALOADER.DALI False DATALOADER.DALI_ON_GPU False DATALOADER.CACHE_EVAL_IMAGES True EVAL_SEGM_NUMPROCS 10 USE_CUDA_GRAPH True EVAL_MASK_VIRTUAL_PASTE True MODEL.BACKBONE.INCLUDE_RPN_HEAD True DATALOADER.NUM_WORKERS 1 PRECOMPUTE_RPN_CONSTANT_TENSORS True DATALOADER.HYBRID True CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION 1