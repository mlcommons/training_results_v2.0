## Steps to launch training on a single node

For single-node training, we use docker to run our container.

### NVIDIA A30 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX A30
single node submission are in the `config_A30_1x2x4.sh` script.

Steps required to launch single node training on NVIDIA A30:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:object_detection .
docker push <docker/registry>/mlperf-nvidia:object_detection
```

2. Launch the training:

```
source config_A30_1x2x4.sh
CONT="<docker/registry>/mlperf-nvidia:object_detection" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v2.0 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA A30 (single node)](#nvidia-a30-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:object_detection .
source config_A30_1x2x4.sh
CONT=mlperf-nvidia:object_detection DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

