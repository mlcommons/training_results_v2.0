## Steps to launch training on a single node

### Nettrix X640 G40 (single node)

Launch configuration and system-specific hyperparameters for the Nettrix X640 G40 single node submission are in the `config_X640_A100_01x16.sh` script.

Steps required to launch single node training on NVIDIA A100-PCIe-80GB:

1. Build docker and prepare dataset

```
# Build a docker using Dockerfile in this directory
nvidia-docker build -t mlperf-nvidia:minigo .

# run docker
nvidia-docker run -v <path/to/store/checkpoint>:/data --rm -it mlperf-nvidia:minigo
cd minigo

# Download dataset, needs gsutil.
# Download & extract bootstrap checkpoint.
gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
tar xfz checkpoint.tar.gz -C ml_perf/

# Download and freeze the target model.
mkdir -p ml_perf/target/
gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/

# comment out L331 in dual_net.py before running freeze_graph.
# L331 is: optimizer = hvd.DistributedOptimizer(optimizer)
# Horovod is initialized via train_loop.py and isn't needed for this step.
CUDA_VISIBLE_DEVICES=0 python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags  --model_path=ml_perf/target/target
mv ml_perf/target/target.minigo ml_perf/target/target.minigo.tf

# uncomment L331 in dual_net.py.
# copy dataset to /data that is mapped to <path/to/store/checkpoint> outside of docker.
# Needed because run_and_time.sh uses the following paths to load checkpoint
# CHECKPOINT_DIR="/data/mlperf07"
# TARGET_PATH="/data/target/target.minigo.tf"
cp -a ml_perf/target /data/
cp -a ml_perf/checkpoints/mlperf07 /data/

# exit docker
```

2. Launch the training:

```
source config_X640_A100_01x16.sh
CONT="mlperf-nvidia:minigo" DATADIR=<path/to/store/checkpoint> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
