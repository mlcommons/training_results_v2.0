## Steps to launch training on a single node

### Nettrix X640 G40 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X640 G40 single node submission are in the `config_X640_A100_01x10.sh` script.

Steps required to launch single node training on NVIDIA A100-PCIe-80GB:

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch .
docker push <docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch
```

2. Launch the training

```
source config_X640_A100_01x10.sh
NV_GPU="0,1,2,3,4,5,6,7,8,9" CONT="<docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> METADATA_DIR=<path/to/metadata/dir> SENTENCEPIECES_DIR=<path/to/sentencepieces/dir> ./run_with_docker.sh
```
