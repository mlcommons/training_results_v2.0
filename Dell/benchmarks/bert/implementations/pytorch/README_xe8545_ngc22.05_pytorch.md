## Steps to launch training on a single node

### Dell PowerEdge XE8545 (single node)
Launch configuration and system-specific hyperparameters for the Dell PowerEdge XE8545
multi node submission are in the following scripts:
* for the 1-node Dell PowerEdge XE8545 submission: `config_XE8545_1x4x56x1.sh`

Steps required to launch multi node training on Dell PowerEdge XE8545:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-dell:language_model .
docker push <docker/registry>/mlperf-dell:language_model
```

2. Launch the training:

1-node Dell PowerEdge XE8545 training:

```
source config_XE8545_1x4x56x1.sh
CONT=mlperf-dell:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1 sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
