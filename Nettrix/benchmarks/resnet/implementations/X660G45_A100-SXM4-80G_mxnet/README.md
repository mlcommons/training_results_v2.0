## Steps to launch training on a single node

### Nettrix X660G45 (single node)

Launch configuration and system-specific hyperparameters for the X660G45 single node submission are in the `config_X660G45_SXM4-80G.sh` script.

Steps required to launch single node training on NVIDIA A100-SXM4-80GB:

1. Build the container and push to a docker registry:

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:image_classification .
docker push <docker/registry>/mlperf-nvidia:image_classification
```

2. Launch the training:

```
source config_X660G45_SXM4-80G.sh
CONT="<docker/registry>/mlperf-nvidia:image_classification" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh

