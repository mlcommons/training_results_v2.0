## Steps to launch training

### NVIDIA DGX A100 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
single node submission are in the `config_DGXA100_001x08x032.sh` script.

Steps required to launch single node training on NVIDIA DGX A100

1. Build the docker container and push to a docker registry

```
docker build --pull -t <DOCKER_REGISTRY>/mlperf-nvidia:single_stage_detector-pytorch .
docker push <DOCKER_REGISTRY>/mlperf-nvidia:single_stage_detector-pytorch
```

2. Launch the training

```
source config_DGXA100_001x08x032.sh
CONT="<DOCKER_REGISTRY>/mlperf-nvidia:single_stage_detector-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
