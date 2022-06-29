# Benchmark

Train [Resnet50](https://github.com/mlcommons/training/tree/master/image_classification) using [Imagenet](http://image-net.org/).

# Software Requirements

* [MosaicML's Composer Library v0.7.0-RC1](https://github.com/mosaicml/composer/tree/v0.7.0-RC1)
* [MosaicML's PyTorch Vision Docker Image](https://hub.docker.com/r/mosaicml/pytorch_vision/tags)
   * Tag: `1.11.0_cu113-python3.9-ubuntu20.04`
   * PyTorch Version: 1.11.0
   * CUDA Version: 11.3
   * Python Version: 1.9
   * Ubuntu Version: 20.04
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# Running Benchmark Configs

1. Launch a Docker container using the `pytorch_vision` Docker Image on your training system.
   
   ```
   docker pull mosaicml/pytorch_vision:1.11.i0_cu113-python3.9-ubuntu20.04
   docker run -it mosaicml/pytorch_vision:1.11.i0_cu113-python3.9-ubuntu20.04
   ``` 
   **Note:** The `mosaicml/pytorch_vision` Docker image can also be used with your container orchestration framework of choice.

1. Download the ImageNet dataset from http://www.image-net.org/.

1. Create the dataset folder and extract training and validation images to the appropriate subfolders.
   The [following script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) can be used to faciliate this process.
   Be sure to note the directory path of where you extracted the dataset.

1. Pick which config you would like to run, te following configs are currently supported:
   
   | `$BENCHMARK_CONFIG` | Relative path to config file |
   | --- | --- |
   | [baseline](../8xA100_80GB-baseline/config.yaml) | `../8xA100_80GB-baseline/config.yaml` |
   | [methods](../8xA100_80GB-methods/config.yaml) | `../8xA100_80GB-methods/config.yaml` |


1. Run the benchmark, you will need to specify a supported `$BENCHMARK_CONFIG` value from the previous step and the path to the dataset.

   ```
   bash ./run_and_time.sh --config $BENCHMARK_CONFIG --datadir <path_to_imagenet_dir> 
   ```
   
   **Note:** The `run_and_time.sh` script sources the `setup.sh` script to setup the environment and invokes `init_dataset.sh` to convert the 
   raw Imagenet dataset to FFCV format.



