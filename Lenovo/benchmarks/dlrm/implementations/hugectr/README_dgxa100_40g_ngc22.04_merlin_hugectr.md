## Steps to launch training on a single node

### NVIDIA DGX A100 40GB (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100 40GB
single node submission are in the `config_DGXA100_40g.sh` script and in the `dgx_a100_40g.py` config file.

To launch the training on a single node with a slurm cluster run:
```
source config_DGXA100_40g.sh
CONT=<docker/registry>/mlperf-nvidia:recommendation_hugectr LOGDIR=<path/to/output/dir> sbatch -N 1 run.sub
```

#### Alternative launch with docker

When generating results for the official v1.0 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX A100 40GB (single node)](#nvidia-dgx-a100-40gb-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
source config_DGXA100_40g.sh
CONT=<docker/registry>mlperf-nvidia:recommendation_hugectr DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
