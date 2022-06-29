## Steps to launch training on a single node with 2xA30

### NVIDIA DGX single node
Launch configuration and system-specific hyperparameters for the NVIDIA A30
multi node submission are in the following scripts:
* for the 2xA30 1-node NVIDIA submission: `config_A30_1x2x224x14.sh`

Steps required to launch multi node training on NVIDIA 2xA30:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

2. Launch the training:

1-node NVIDIA 2xA30 training:

```
source config_A30_1x2x224x14.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1 sbatch -N $DGXNNODES -t $WALLTIME run.sub
```