# Graphcore: packed BERT 

This readme dscribes how to run BERT on IPUs.

## Wikipedia datasets
Follow the `bert_data/README.md` to construct the packed BERT Wikipedia dataset for Mlperf.

## Pre-trained checkpoint
Follow the steps in the Mlcommons/Mlperf reference implementation 2.0.0 to download the tf1 checkpoint. 
Place the files in `tf-checkpoints`. More specifically the config files will expect to load the checkpoint from `tf-checkpoints/bs64k_32k_ckpt/model.ckpt-28252`.

## Prepare the environment

### 1) Download the Poplar SDK
  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for poplar and popART.

### 2) Compile `custom_ops`

Compile custom_ops:

```bash
make
```

This should create `custom_ops.so`.

### 3) Prepare Python3 virtual environment

Create a virtualenv and install the required packages:

```bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

## Running BERT

Pod16 and 64 runs can be launched directly using `bert.py`:
```
python bert.py --config=./configs/podXX-XXX.json
```

Pod128 and 256 require runs to be launched using `poprun` with the following environment variables:

```
export HOSTS=<PROVIDED_HOST_IPS>
export VIPU_SERVER_HOST=<PROVIDED_VIPU_SERVER_HOST>
export PARTITION_NAME=<PROVIDED_PARTITION_NAME>
export CLUSTER_NAME=<PROVIDED_CLUSTER_NAME>
export TCP_IF_INCLUDE=<PROVIDED_NETWORK_INTERFACES>
```

Additionally, one must also define the following for pod128
```
export NUM_INSTANCE=2
export NUM_REPLICAS=16
export NUM_ILDS=2
export IPUS_PER_REPLICA=8
```

Likewise, for pod256:
```
export NUM_INSTANCE=4
export NUM_REPLICAS=32
export NUM_ILDS=4
export IPUS_PER_REPLICA=8
```

Pod128 or 256 runs can then be launched using the following command:
```bash
poprun -vv  --remove-partition=0 --num-instances {NUM_INSTANCE} --num-replicas {NUM_REPLICAS}  --num-ilds={NUM_ILDS} --ipus-per-replica {IPUS_PER_REPLICA} --numa-aware=yes --host {HOSTS} --vipu-server-host={VIPU_SERVER_HOST} --vipu-server-timeout=600 --vipu-partition={PARTITION_NAME} --vipu-cluster={CLUSTER_NAME} --reset-partition=no --update-partition=no --mpi-global-args='--tag-output  --allow-run-as-root  --mca oob_tcp_if_include {TCP_IF_INCLUDE} --mca btl_tcp_if_include {TCP_IF_INCLUDE}' --mpi-local-args='-x SHARED_EXECUTABLE_CACHE -x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=OFF -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=OFF -x POPART_LOG_LEVEL=OFF -x GCL_LOG_LEVEL=OFF' python bert.py --config=configs/podXXX-XXX.json
```

During the first run, an executable will be compiled and saved for subsequent runs.

