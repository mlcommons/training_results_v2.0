export NEXP=40
source config_XIANGXUE-3B_A100-PCIE-40GBx16.sh
CONT=mlperf-nvidia:image_segmentation-mxnet DATADIR=/dev/shm/unet3d_data LOGDIR=$(pwd)/results ./run_with_docker.sh
