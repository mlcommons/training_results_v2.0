## Steps to launch training on a single node

### Nettrix X640G40 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X640G40 single node submission are in the `config_X640_A30.sh` script.

To launch the trainining on a single node:
```
source config_X640_A30.sh
NV_GPU="0,1,2,3,4,5,6,7" CONT=<docker/registry>/mlperf-nvidia:recommendation_hugectr LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
