## Steps to launch training

### NVIDIA A30 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA A30
single node submission are in the `config_A30_1x2x128x2.sh` script.

Steps required to launch single node training on NVIDIA A30

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch .
docker push <docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch
```

2. Launch the training

```
source config_A30_1x2x128x2.sh
CONT="<docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> METADATA_DIR=<path/to/metadata/dir> SENTENCEPIECES_DIR=<path/to/sentencepieces/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
