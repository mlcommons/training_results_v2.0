
## Bert TPU-v4-8192: Instructions to run

### Dataset preparation

Follow the instructions to create TFRecords in the following link.

*   [BERT Wikipedia dataset preparation](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets)

### Create the TPU VM
```
$ gcloud alpha compute tpus tpu-vm create $TPU_NAME --accelerator-type=v4-8192 --version=v2-alpha-tpuv4-pod
```

### Setup Network-attached disk with datasets

Create disks:

```
$ gcloud compute disks create $DISK_NAME  --type=pd-ssd
```

Attach disks:

```
$ gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME --disk=$DISK_NAME --mode=read-write
```


### Setup the VMs

Mount disks on all VMs:

```
$ gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all \
  --command="sudo mkdir -p /mnt/disks/persist && sudo mount -o ro,noload /dev/sdb && sudo blockdev --setra 261244 /dev/sdb"
```

Set up the TPU runtimes:

```
$ echo '[Unit]
Description=TPU Runtime & grpc server
After=docker.service

[Service]
Environment="HOME=/home/tpu-runtime"
EnvironmentFile=/home/tpu-runtime/tpu-env
ExecStartPre=/bin/bash -c "/usr/bin/systemctl set-environment TF_DOCKER_URL=gcr.io/cloud-tpu-v2-images/grpc_tpu_worker_v4:mlperf-2.0"
ExecStartPre=/bin/bash -c "/usr/bin/systemctl set-environment HOST_IP=$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip' -H 'Metadata-Flavor: Google')"
ExecStartPre=/bin/mkdir -p /tmp/tflogs
ExecStartPre=/bin/chmod +r /tmp/tflogs
ExecStartPre=/bin/chown tpu-runtime:tpu-runtime /tmp/tflogs
ExecStartPre=/usr/bin/docker-credential-gcr configure-docker
ExecStartPre=-/usr/bin/docker rm -f -v tpu-runtime
ExecStartPre=/usr/bin/docker pull $TF_DOCKER_URL
ExecStartPre=/bin/bash /var/scripts/container-restartCount.sh tpu-runtime
ExecStart=/usr/bin/docker run --net=host --rm --name=tpu-runtime -v /tmp/profiles:/profiles/ -v /mnt/disks/persist:/mnt/disks/persist -v /tmp/tflogs:/tmp --user=2000:2000 --ulimit=memlock=68719476736 --privileged=true -e MLPERF_SUBMISSION_CONFIG=bert_v4-8192 $TF_DOCKER_URL --tpu_hostname_override=${HOST_IP} --envelope_enabled=false
ExecStop=/usr/bin/docker stop tpu-runtime
ExecStopPost=/usr/bin/docker rm -v tpu-runtime
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target' | tee tpu-runtime.service

$ gcloud alpha compute tpus tpu-vm scp tpu-runtime.service $TPU_NAME: --worker=all

$ gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all \
  --command="sudo mv tpu-runtime.service /etc/systemd/system/tpu-runtime.service && sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime"

```
* Note that `gcr.io/cloud-tpu-v2-images` is the official GCR hub for all official CloudTPU images, including images like `v2-alpha-tpuv4-pod, tpu-vm-tf-2.8.0-pod`.

### Copy the code to the VM

Copy the source code, setup and run scripts and the contrib.tar.gz artifact to the VM.

From the Google top-level submission dir:
```
gcloud alpha compute tpus tpu-vm scp benchmarks/bert/implementations/bert-cloud-tpu-v4-8192-TF $TPU_NAME:bert --recurse
gcloud alpha compute tpus tpu-vm scp benchmarks/bert/implementations/cloud-tpu-v4-8192-TF/setup.sh $TPU_NAME:
gcloud alpha compute tpus tpu-vm scp benchmarks/bert/implementations/cloud-tpu-v4-8192-TF/run_bert_v4-8192.sh $TPU_NAME:
gcloud alpha compute tpus tpu-vm scp benchmarks/contrib.tar.gz $TPU_NAME:
```

Set `$PROJECT_NAME` and `$BUCKET` environment variables to access data in GCP.
Alternative set $LOCAL_DATA_PATH if data is in a local drive (e.g. SSD).
Eventually data will be pointed to by `$MLP_BERT_DATA` which depends on either
`$LOCAL_DATA_PATH` or `$BUCKET`

Possibly set `$TOP_LEVEL_DIR` to the directory above your model. By default this is set to nothing, so everything runs out of the home dir.
All the above environment variables are set in `setup.sh` and `run_bert_v4-8192.sh`.

### Inside the VM

SSH to the VM

```
$ gcloud alpha compute tpus tpu-vm ssh $TPU_NAME
```

Run the following scripts:

```
$ cd $TOP_LEVEL_DIR

$ tar xvf contrib.tar.gz

$ source setup.sh

$ ./run_bert_v4_8192.sh
```

### Possible issues

If the gcloud commands complain for missing PROJECT_ID / zone, you need to add `--project=<GCP TPU-VM PROJECT ID> --zone=us-central2-b` to the command line.

You may need to run `gcloud auth login --update-adc` the first time you log into the VM.

If you are using local SSD storage and you cannot attach a read-write disk, you will need to copy the data set in advance and then attach the disk as `read-only`.

If copying and restarting the TPU runtimes fails for some workers you will need to re-execute the failing commands just for the worker(s) that failed.
For example, if copying tpu-runtime.service fails on worker 11, then you will need to re-execute this command, setting `--worker=11` instead of `worker=all`.

```$ gcloud alpha compute tpus tpu-vm scp tpu-runtime.service $TPU_NAME: --worker=all```
