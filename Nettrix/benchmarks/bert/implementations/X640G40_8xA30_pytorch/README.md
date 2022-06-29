### Steps to launch training on a single node

### Nettrix X640G40 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X640G40 single node submission are in the `config_X640_A30.sh` script.

Steps required to launch single node training on NVIDIA A30:

1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

2. Launch the training:
```
source config_X640_A30.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1> ./run_with_docker.sh

