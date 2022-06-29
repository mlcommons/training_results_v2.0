
## SSD / RetinaNet TPU-v4-4096 Instructions to run

### Prepare checkpoint and dataset:

To translate the checkpoint from PyT to TF: Use script ```convert_chekpoint.py```
from ```benchmarks/ssd/implementations/ssd-cloud-tpu-v4-4096-TF/```

```
python convert_checkpoint.py --pyt_pickle_file <path_input> --tf_checkpoint_file <path_output>
```

The data set download instructions are the same as the reference.

We use a script to convert the downloaded data set into TFRecord format which is not part of the MLPerf code.
Instead, we use open-source TF code. The steps are:

```
git clone https://github.com/tensorflow/models.git tf-models
git clone https://github.com/tensorflow/tpu.git

PYTHONPATH="tf-models:tf-models/research" python3 tpu/tools/datasets/create_coco_tf_record.py \
  --image_dir=<path to training data> \
  --object_annotations_file=<path to label json file> \
  --output_file_prefix=/mnt/data/openimages_tf_1024_shard/train \
  --num_shards=1024
```

### Create the TPU VM
```
$ gcloud alpha compute tpus tpu-vm create $TPU_NAME --accelerator-type=v4-4096 --version=v2-alpha-tpuv4-pod
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
ExecStart=/usr/bin/docker run --net=host --rm --name=tpu-runtime -v /tmp/profiles:/profiles/ -v /mnt/disks/persist:/mnt/disks/persist -v /tmp/tflogs:/tmp --user=2000:2000 --ulimit=memlock=68719476736 --privileged=true -e MLPERF_SUBMISSION_CONFIG=ssd_v4-4096 $TF_DOCKER_URL --tpu_hostname_override=${HOST_IP} --envelope_enabled=false
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
gcloud alpha compute tpus tpu-vm scp benchmarks/ssd/implementations/ssd-cloud-tpu-v4-4096-TF $TPU_NAME:ssd --recurse
gcloud alpha compute tpus tpu-vm scp benchmarks/ssd/implementations/cloud-tpu-v4-4096-TF/setup.sh $TPU_NAME:
gcloud alpha compute tpus tpu-vm scp benchmarks/ssd/implementations/cloud-tpu-v4-4096-TF/run_ssd_v4-4096.sh $TPU_NAME:
gcloud alpha compute tpus tpu-vm scp benchmarks/contrib.tar.gz $TPU_NAME:
```

Set `$PROJECT_NAME` and `$BUCKET` environment variables to access data in GCP.
Alternatively set `$LOCAL_DATA_PATH` if data is in a local drive (e.g. SSD).
Eventually data will be pointed to by `$MLP_SSD_DATA` which depends on either
```$LOCAL_DATA_PATH``` or `$BUCKET`.

Possibly set `$TOP_LEVEL_DIR` to the directory above your model. By default this is set to nothing, so everything runs out of the home dir.
All the above environment variables are set in `setup.sh` and `run_ssd_v4-4096.sh`.

### Inside the VM:

```
$ gcloud alpha compute tpus tpu-vm ssh $TPU_NAME
```

#### Build the CPP/PyCLIF code:

Using the official CLIF [Dockerfile](https://github.com/google/clif/blob/main/Dockerfile):
```
$ sudo docker build --build-arg=UBUNTU_VERSION=20.04 - < Dockerfile
```

Log into the container:
```
$ sudo docker run -v $(pwd)/staging:/staging --rm -it <image> /bin/bash
```

Run some initial setup:
```
$ update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 10

$ apt-get update
$ apt-get install git
$ apt-get install libopencv-dev
$ apt-get install libjsoncpp-dev
$ apt-get install python-dev
$ pip install google-api-python-client
$ git clone https://github.com/google/clif.git
$ ./clif/INSTALL.sh
```
If you get a parsing error while running clif/INSTALL.sh, then run the following:
```
pip uninstall pyparsing
pip install pyparsing==2.2
```
and then delete and recreate the clif git repository, as above.

```
$ pip install numpy
$ ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2
```

Build the packages:
1. Copy the CMakeLists.txt from Google's submission to file clif/examples.

```
$ cp $TOP_LEVEL_DIR/ssd/ssd-cloud-tpu-v4-4096-TF/CMakeLists.txt ./clif/examples/.
```

2. Add the pycocotools subdirectory under clif/examples.

```
$ cp -r $TOP_LEVEL_SIR/ssd/ssd-cloud-tpu-v4-4096-TF/pycocotools ./clif/examples/.
```

3. Add the following lines to CMakeLists.txt

```
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory("pycocotools")
```

#### Run the following from the clif/examples directory:
```
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 .
make
pip install .
```

#### Run the workload

```
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjsoncpp.so:/usr/lib/x86_64-linux-gnu/libopencv_core.so:/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so

$ cd $TOP_LEVEL_DIR

$ tar xvf contrib.tar.gz

$ source setup.sh

$ ./run_ssd_v4-4096.sh
```


### Possible issues

If the gcloud commands complain for missing PROJECT_ID / zone, you need to add `--project=<GCP TPU-VM PROJECT ID> --zone=us-central2-b` to the command line.

You may need to run `gcloud auth login --update-adc` the first time you log into the VM.

If you are using local SSD storage and you cannot attach a read-write disk, you will need to copy the data set in advance and then attach the disk as `read-only`.

If copying and restarting the TPU runtimes fails for some workers you will need to re-execute the failing commands just for the worker(s) that failed.
For example, if copying tpu-runtime.service fails on worker 11, then you will need to re-execute this command, setting `--worker=11` instead of `worker=all`.

```$ gcloud alpha compute tpus tpu-vm scp tpu-runtime.service $TPU_NAME: --worker=all```
