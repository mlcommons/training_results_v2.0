## Steps to launch training on a single node

### Nettrix X660G45L (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X660G45L single-node submission are in the `config_X660G45L.sh` script.

Steps required to launch single-node training on NVIDIA A100-SXM4-80GB:

1. Build the docker container and push to a docker registry

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:image_segmentation-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_segmentation-mxnet
```

2. Launch the training

```
source config_X660G45L.sh
CONT="<docker/registry>/mlperf-nvidia:image_segmentation-mxnet" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
