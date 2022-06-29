## Steps to launch training on a single node

### Nettrix X640 G40 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X640 G40 single node submission are in the `config_X640_A100_01x10.sh` script.

Steps required to launch single node training on NVIDIA A100-PCIe-80GB:

1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:object_detection .
docker push <docker/registry>/mlperf-nvidia:object_detection
```

2. Launch the training:

```
source config_X640_A100_01x10.sh
NV_GPU="0,1,2,3,4,5,6,7,8,9" TORCH_HOME=<path/to/torch/dir> CONT="<docker/registry>/mlperf-nvidia:object_detection" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh

```
