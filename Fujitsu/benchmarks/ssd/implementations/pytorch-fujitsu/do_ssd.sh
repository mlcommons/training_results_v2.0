
source config_GX2570M6.sh
CONT=nvcr.io/nvdlfwea/mlperfv20/ssd:20220509.pytorch
DATADIR=/mnt/data4/work/ssd-openimages
LOGDIR=$(realpath ../logs-ssd)
TORCH_HOME=$(realpath ./torch-model-cache)
NEXP=1
num_of_run=5

for idx in $(seq 1 $num_of_run); do
    CONT=$CONT DATADIR=$DATADIR LOGDIR=$LOGDIR TORCH_HOME=$TORCH_HOME NEXP=$NEXP bash run_with_docker.sh
done
