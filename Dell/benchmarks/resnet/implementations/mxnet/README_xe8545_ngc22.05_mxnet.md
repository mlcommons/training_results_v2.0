## Steps to launch training

### Dell PowerEdge XE8545 (single node)

Launch configuration and system-specific hyperparameters for the Dell PowerEdge XE8545
single node submission are in the `config_XE8545.sh` script.

Steps required to launch single node training on Dell PowerEdge XE8545:

1. Build the container and push to a docker registry:

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-dell:image_classification .
docker push <docker/registry>/mlperf-dell:image_classification
```

2. Launch the training:

```
source config_XE8545.sh
CONT="<docker/registry>/mlperf-dell:image_classification" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v2.0 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
cd ../implementations/mxnet
docker build --pull -t mlperf-dell:image_classification .
source config_XE8545.sh
CONT=mlperf-dell:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
