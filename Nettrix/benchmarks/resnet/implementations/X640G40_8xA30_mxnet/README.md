## Steps to launch training on a single node

### Nettrix X640G40 (single node)

Launch configuration and system-specific hyperparameters for the X640G40 single node submission are in the `config_X640_A30.sh` script.

Steps required to launch single node training on NVIDIA A30:

1. Build the container and push to a docker registry:

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:image_classification .
docker push <docker/registry>/mlperf-nvidia:image_classification
```

2. Launch the training:

```
source config_X640_A30.sh
CONT="<docker/registry>/mlperf-nvidia:image_classification" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
