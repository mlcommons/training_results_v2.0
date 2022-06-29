# Index
1 - Setup
1.1 - Install firmware, driver, SynapseAI 1.4.0
1.2 - Build and deploy HabanaLabs MLPERF training 2.0 container in the cluster
2 - Running ResNet50
2.1 - Prepare Imagenet dataset
2.2 - Run ResNet50 at scale in the cluster




1 - Setup

1.1 - Install firmware, driver, SynapseAI 1.4.0

Follow the steps in [Setup and Install](https://docs.habana.ai/en/v1.4.0/Installation_Guide/GAUDI_Installation_Guide.html) to setup each compute node in the cluster.


1.2 - Build and deploy HabanaLabs MLPERF training 2.0 container in the cluster

For each compute node, do 
    git clone HabanaLabs MLPERF training 2.0 code from public repo at https://github.com/mlcommons/
    pull HabanaLabs release container 1.4.0 from vault at vault.habana.ai/gaudi-docker/1.4.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.8.0:1.4.0-442
    build MLPERF training 2.0 container by 
        copying MLPERF training 2.0 code to /root/kerasWork
        copying ssh keys to enable passwordless ssh to /root/.ssh/
        copying hostfile that contains a list of hosts in the cluster to /root/shared/
        installing numactl package
        naming the container mlperf2.0

    start MLPERF training 2.0 container by executing
        docker run --privileged --security-opt seccomp=unconfined \
           --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
           --name $CONTAINER_NAME -d --rm               \
           -e DISPLAY=$DISPLAY                          \
           -e LOG_LEVEL_ALL=6                           \
           -v /sys/kernel/debug:/sys/kernel/debug       \
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro          \
           -v $RESULTS_DIR:/root/scratch                \
           -v ${IMAGENET_HOST}:${IMAGENET_CONTAINER}    \
           -v ${MLPERF_HOST}:${MLPERF_CONTAINER}        \
           --cap-add=sys_nice --cap-add=SYS_PTRACE      \
           --user root --workdir=/root --net=host       \
           --ulimit memlock=-1:-1 ${DOCKER_IMAGE} sleep infinity

        docker exec mlperf2.0 bash -c "service ssh start"

        where:
            $RESULTS_DIR path to results directory in the host
            $DATASET_DIR path to workload dataset directory in the host
            gaudinet.json contains the list of gaudi network ports
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



2 - Running Resnet50

2.1 - Prepare Imagenet dataset

## Dataset Preparation

[ImageNet dataset preparation](https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment)                                                     


2.2 - Running ResNet50 at scale in the cluster

Log into one of the mlperf2.0 containers
Given a runtime configuration, for instance, 8 gaudis run
    cd /root/kerasWork/Habana/benchmarks/resnet/implementations/HLS-1H-N2
    edit defaults.cfg with the right location of your dataset tf records inside the container
        for example, IMAGENET_DIR=/root/datasets/imagenet/tf_records
    execute the script launch_keras_resnet_hvd.sh for a cluster run based on hostfile
    It will place the results of the run at $RESULTS_DIR/resnet in the host.
