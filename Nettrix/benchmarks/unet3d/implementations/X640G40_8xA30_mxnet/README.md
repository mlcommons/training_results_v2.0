## Steps to launch training on a single node

### Nettrix X640G40 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X640G40 single-node submission are in the `config_X640_A30.sh` script.

Steps required to launch single-node training on NVIDIA A30:

1. Build the docker container and push to a docker registry

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:image_segmentation-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_segmentation-mxnet
```

2. Launch the training

```
source config_X640_A30.sh
CONT="<docker/registry>/mlperf-nvidia:image_segmentation-mxnet" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
