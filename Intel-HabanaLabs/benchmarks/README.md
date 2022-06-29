# Habana MLPerf training 2.0 submisssion

## Install firmware, driver, SynapseAI 1.4.99

Follow the steps in [Setup and Install](https://docs.habana.ai/en/v1.4.0/Installation_Guide/GAUDI_Installation_Guide.html) to setup each compute node in the cluster.


## Build and deploy HabanaLabs MLPERF training 2.0 container in the cluster

For each compute node, do
```
git clone HabanaLabs MLPERF training 2.0 code from public repo at https://github.com/mlcommons/
```

Pull HabanaLabs release container 2.0.0 from vault at:

```
docker pull vault.habana.ai/gaudi-docker-mlperf/ver2.0/tensorflow-installer-tf-cpu-2.8.0:1.4.99-42
```

Build MLPERF training 2.0 container by

1. Copying MLPERF training 2.0 code to /root/MLPERF
2. Copying ssh keys to enable passwordless ssh to /root/.ssh/
3. Creating hostfile that contains a list of hosts in the cluster to /root/share

    (e.g., for single node: ```echo your-machine-ip > /root/shared/hosts```)
4. Installing numactl package (required for large scale Gaudi)
5. Naming the container mlperf2.0_img

For Gaudi2 start MLPERF training 2.0 container by executing

```
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf2.0 -td                         \
           --device=/dev/hl_controlD0:/dev/hl_controlD0 \
           --device=/dev/hl_controlD1:/dev/hl_controlD1 \
           --device=/dev/hl_controlD2:/dev/hl_controlD2 \
           --device=/dev/hl_controlD3:/dev/hl_controlD3 \
           --device=/dev/hl_controlD4:/dev/hl_controlD4 \
           --device=/dev/hl_controlD5:/dev/hl_controlD5 \
           --device=/dev/hl_controlD6:/dev/hl_controlD6 \
           --device=/dev/hl_controlD7:/dev/hl_controlD7 \
           --device=/dev/hl0:/dev/hl0                   \
           --device=/dev/hl1:/dev/hl1                   \
           --device=/dev/hl2:/dev/hl2                   \
           --device=/dev/hl3:/dev/hl3                   \
           --device=/dev/hl4:/dev/hl4                   \
           --device=/dev/hl5:/dev/hl5                   \
           --device=/dev/hl6:/dev/hl6                   \
           --device=/dev/hl7:/dev/hl7                   \
           -e DISPLAY=$DISPLAY                          \
           -e LOG_LEVEL_ALL=6                           \
           -v /sys/kernel/debug:/sys/kernel/debug       \
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro          \
           -v /tmp:/tmp                                 \
           -v $LOG_DIR:/root/scratch                    \
           -v $DATASET_DIR:/root/datasets/              \
           --cap-add=sys_nice --cap-add=SYS_PTRACE      \
           --user root --workdir=/root --net=host       \
           --ulimit memlock=-1:-1 mlperf2.0_img

docker exec mlperf2.0 bash -c "service ssh start"
```

for Gaudi start MLPERF training 2.0 container by executing

```
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf2.0 -td                         \
           --device=/dev/hl_controlD0:/dev/hl_controlD0 \
           --device=/dev/hl_controlD1:/dev/hl_controlD1 \
           --device=/dev/hl_controlD2:/dev/hl_controlD2 \
           --device=/dev/hl_controlD3:/dev/hl_controlD3 \
           --device=/dev/hl0:/dev/hl0                   \
           --device=/dev/hl1:/dev/hl1                   \
           --device=/dev/hl2:/dev/hl2                   \
           --device=/dev/hl3:/dev/hl3                   \
           -e DISPLAY=$DISPLAY                          \
           -e LOG_LEVEL_ALL=6                           \
           -v /sys/kernel/debug:/sys/kernel/debug       \
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro          \
           -v /tmp:/tmp                                 \
           -v /etc/gaudinet.json:/etc/gaudinet.json     \
           -v $LOG_DIR:/root/scratch                    \
           -v $DATASET_DIR:/root/datasets/              \
           --cap-add=sys_nice --cap-add=SYS_PTRACE      \
           --user root --workdir=/root --net=host       \
           --ulimit memlock=-1:-1 mlperf2.0_img

docker exec mlperf2.0 bash -c "service ssh start"
```
Where:

 1. ```$LOG_DIR``` is the path to the results directory on the host

 2. ```$DATASET_DIR``` is the path to the workload dataset directory on the host

The ```/etc/gaudinet.json``` file contains the list of gaudi network ports to control the connectivity between HLS-1H

```
            {
                "NIC_NET_CONFIG":[
                    {
                        "NIC_MAC":"",
                        "NIC_IP":"",
                        "SUBNET_MASK":"",
                        "GATEWAY_MAC":""
                    },
                    ...
                    {
                        "NIC_MAC":"",
                        "NIC_IP":"",
                        "SUBNET_MASK":"",
                        "GATEWAY_MAC":""
                    }
                ]
             }
```


# Resnet50
## Prepare Imagenet dataset

 1. Sign up with [image-net.org](http://image-net.org/download-images) and acquire the rights to download original images
 2. Follow the link to the 2012 ILSVRC and download ILSVRC2012_img_val.tar and ILSVRC2012_img_train.tar
 3. Use the script below to unpact the dataset. Set IMAGENET_HOME to a folder where dataset should be placed

```
export IMAGENET_HOME=/path/to/imagenet
mkdir -p $IMAGENET_HOME/val
mkdir -p $IMAGENET_HOME/train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/val
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
cd $IMAGENET_HOME/train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
rm $IMAGENET_HOME/train/*.tar
cd $IMAGENET_HOME/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## Run and time
Inside docker install additional packages required for Resnet50 and extend PYTHONPATH:
```
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
export PYTHONPATH=$RESNET_IMPLEMENTATIONS/HLS-Gaudi2/mlperf-logging:$PYTHONPATH
```
Execute the script
```
cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2
./launch_keras_resnet_hvd.sh --config batch_256.cfg --cpu-pin cpu --data-dir /path/to/imagenet --jpeg-data-dir /path/to/imagenet
```
for a cluster run based on hostfile.
Use the ```$IMAGENET_HOME``` folder from previous step for ```--data-dir``` and ```--jpeg-data-dir```.
Results of the run will be places in $LOG_DIR on the host.


# Bert

## Prepare packed wiki dataset

**Location to download Dataset and Checkpoint:** [Dataset and Checkpoint download location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)

**Dataset Preparation:** In order to use dataset one needs to preprocess it similarly as it described in [Bert dataset preparation](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets).

Each of 500 dataset files can be converted in the following way:
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
pip3 install -r requirements.txt
python3 pretraining/create_pretraining_data.py \
    --input_file=<path to downloaded and unzipped dataset>/part-00XXX-of-00500 \
    --output_file=<output dir for tfrecord files>/part-00XXX-of-00500 \
    --vocab_file=<path to downloaded vocab.txt> \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=10
```


After tfrecord files are ready we pack them using similar code as suggested by [GraphCore for v1.0 submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data)

```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
pip3 install -r requirements.txt
python3 pack_pretraining_data_tfrec.py \
    --input-glob /path-to-tfrecords-dir \
    --output-dir /path-to-output-dir \
    --max-files 500
```

For additional details please refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027).

## Run and time

Log into one of the mlperf2.0 containers

Given a runtime configuration, for instance, 128 gaudis run
```
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-1H-N32
```
Edit defaults.cfg with the right location of your packed dataset tf records inside the container

for example, ```INPUT_FILES_DIR_PACKED=/root/datasets/bert_pretraining/packed```
execute the script ```launch_bert_hvd.sh --config defaults.cfg``` for a cluster run based on hostfile
It will place the results of the run at $LOG_DIR on the host.
