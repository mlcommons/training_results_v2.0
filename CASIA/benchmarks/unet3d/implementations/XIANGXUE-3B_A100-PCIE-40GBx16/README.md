## Steps to launch training

### XIANGXUE-3B_A100-PCIE-40GBx16_7773X (single node)

Launch configuration and system-specific hyperparameters for the XIANGXUE-3B_A100-PCIE-40GBx16_7773X
single-node submission are in the `config_XIANGXUE-3B_A100-PCIE-40GBx16.sh` script.

Steps required to launch single-node training on XIANGXUE-3B_A100-PCIE-40GBx16_7773X

1. Build the docker container and push to a docker registry

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:image_segmentation-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_segmentation-mxnet
```

2. Launch the training

#### Alternative launch with nvidia-docker

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
bash unet3d_runner_16GPUs.sh
```