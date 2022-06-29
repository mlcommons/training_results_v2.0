### Steps to download data
```
./init_datasets.sh
```

## Steps to launch training

### FUJITSU PRIMERGY GX2570 M6 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2570 M6
single node submission are in the `config_DGXA100.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2570 M6:

```
cd ../mxnet-fujitsu
docker build --pull -t mlperf-fujitsu:image_classification .
source config_DGXA100.sh
CONT=mlperf-fujitsu:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
