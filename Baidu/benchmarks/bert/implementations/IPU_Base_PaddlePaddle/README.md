# Baidu: packed BERT with PaddlePaddle

## Overview

BERT (Bi-directional Encoder Representations for Transfomers) is a Deep Learning NLP model used to create embeddings for words that are based on the context in which they appear in a sentence. These embeddings can be used for a number of downstream tasks such as sentiment detection and question-answering.

## Wikipedia datasets

Follow the `bert_data/README.md` to construct the packed BERT Wikipedia dataset for Mlperf.

## Pre-trained checkpoint

Follow the steps in the Mlcommons/Mlperf reference implementation 2.0.0 to download the tf1 checkpoint.

## Quick Start Guide

### Prepare

Assuming that the packed dataset and checkpoint is placed in `${WORKPLACE}`.

```bash
# build docker image
docker build -t mlperf_paddle_bert:latest .

# ipuof.conf
# In order to communicate with the IPU-POD, the configuration file `ipuof.conf` with details of how to connect to the IPU-Machines is required.
# If already have the ipuof.conf, please move it to `${WORKPLACE}`.
# else, please create ipuof.conf by following the commands as below:
# Show IPU-POD partitions:
vipu list partitions
# Choose one of partitions to fill in `${PARTITION_NAME}` and create ipuof.conf
vipu-admin get partition ${PARTITION_NAME} --gcd 0 --ipuof-configs > ipuof.conf
# Then, move it to `${WORKPLACE}`

# create container
docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
--device=/dev/infiniband/ --ipc=host --name paddle-mlperf-v2.0 \
-v ${WORKPLACE}:/paddle_develop \
-e IPUOF_CONFIG_PATH=/paddle_develop/ipu.conf \
-it mlperf_paddle_bert:latest bash
```

`All of additional steps are required to be executed in the container.`

### Execution

`POD16:`

```
python create_submission.py 16
```

`POD64:`

```
python create_submission.py 64
```

The results will be saved in `results/`.
