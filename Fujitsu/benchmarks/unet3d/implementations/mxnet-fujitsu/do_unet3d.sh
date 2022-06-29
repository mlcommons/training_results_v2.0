container_name=nvcr.io/nvdlfwea/mlperfv20/unet3d:20220509.mxnet
data_dir=/mnt/data4/work/3d-unet/data-dir
result_dir=$(realpath ../logs-unet3d)

source config_GX2570M6.sh
num_of_run=40

for idx in $(seq 1 $num_of_run); do
    CONT=$container_name DATADIR=$data_dir LOGDIR=$result_dir NEXP=1 ./run_with_docker.sh
done
