
## MaskRCNN- TPU-v4-2048: Instructions to run

### Dataset preparation

MaskRCNN iuses the COCO dataset.

Clone this repository from git: git clone https://github.com/tensorflow/tpu/

Script `tpu/tools/datasets/download_and_preprocess_coco.sh` will convert the COCO dataset into a set of TFRecords that our trainer expects.

For example:
```
bash download_and_preprocess_coco.sh ./data/dir/coco
```

This will install the required libraries and then run the preprocessing script. It outputs a number of *.tfrecord files in your data directory.

Follow this link for more information:

### Create the TPU VM
```
$ gcloud alpha compute tpus tpu-vm create $TPU_NAME --accelerator-type=v4-2048 --version=v2-alpha-tpuv4-pod
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
ExecStart=/usr/bin/docker run --net=host --rm --name=tpu-runtime -v /tmp/profiles:/profiles/ -v /mnt/disks/persist:/mnt/disks/persist -v /tmp/tflogs:/tmp --user=2000:2000 --ulimit=memlock=68719476736 --privileged=true -e MLPERF_SUBMISSION_CONFIG=maskrcnn_v4-2048 $TF_DOCKER_URL --tpu_hostname_override=${HOST_IP} --envelope_enabled=false
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
gcloud alpha compute tpus tpu-vm scp benchmarks/maskrcnn/implementations/maskrcnn-cloud-tpu-v4-2048-TF $TPU_NAME:maskrcnn --recurse
gcloud alpha compute tpus tpu-vm scp benchmarks/maskrcnn/implementations/cloud-tpu-v4-2048-TF/setup.sh $TPU_NAME:
gcloud alpha compute tpus tpu-vm scp benchmarks/maskrcnn/implementations/cloud-tpu-v4-2048-TF/run_maskrcnn_v4-2048.sh $TPU_NAME:
gcloud alpha compute tpus tpu-vm scp benchmarks/contrib.tar.gz $TPU_NAME:
```

Set `$PROJECT_NAME` and `$BUCKET` environment variables to access data in GCP.
Alternatively set `$LOCAL_DATA_PATH` if data is in a local drive (e.g. SSD).
Eventually data will be pointed to by `$MLP_MASKRCNN_DATA` which depends on either
`$LOCAL_DATA_PATH` or `$BUCKET`.

Possibly set `$TOP_LEVEL_DIR` to the directory above your model. By default this is set to nothing, so everything runs out of the home dir.
All the above environment variables are set in `setup.sh` and `run_maskrcnn_v4-2048.sh`.

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
$ cp $TOP_LEVEL_DIR/maskrcnn/maskrcnn-cloud-tpu-v4-2048-TF/CMakeLists.txt ./clif/examples/.
```

2. Add the pycocotools and postprocess subdirectory under clif/examples.

```
$ cp -r $TOP_LEVEL_DIR/maskrcnn/maskrcnn-cloud-tpu-v4-2048-TF/pycocotools ./clif/examples/.
$ cp -r $TOP_LEVEL_DIR/maskrcnn/maskrcnn-cloud-tpu-v4-2048-TF/postprocess ./clif/examples/.
```

3. Add the following lines to CMakeLists.txt

```
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory("pycocotools")
add_subdirectory("postprocess")
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

$ ./run_maskrcnn_v4-2048.sh
```

### Possible issues

If the gcloud commands complain for missing PROJECT_ID / zone, you need to add `--project=<GCP TPU-VM PROJECT ID> --zone=us-central2-b` to the command line.

You may need to run `gcloud auth login --update-adc` the first time you log into the VM.

If you are using local SSD storage and you cannot attach a read-write disk, you will need to copy the data set in advance and then attach the disk as `read-only`.

If copying and restarting the TPU runtimes fails for some workers you will need to re-execute the failing commands just for the worker(s) that failed.
For example, if copying tpu-runtime.service fails on worker 11, then you will need to re-execute this command, setting `--worker=11` instead of `worker=all`.

```$ gcloud alpha compute tpus tpu-vm scp tpu-runtime.service $TPU_NAME: --worker=all```
