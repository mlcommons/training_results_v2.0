## Steps to launch training on a single node

### FUJITSU PRIMERGY GX2570M6 (single node)
Launch configuration and system-specific hyperparameters for the PRIMERGY GX2570M6
single node submission are in the following scripts:
* for the 1-node PRIMERGY GX2570M6 submission: `config_GX2570M6.sh`

Steps required to launch training on PRIMERGY GX2570M6:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-fujitsu:single_stage_detector .
docker push <docker/registry>/mlperf-fujitsu:single_stage_detector
```

2. Launch the training:

1-node PRIMERGY GX2570M6 training:

```
cd ../pytorch-fujitsu
bash do_ssd.sh
```
