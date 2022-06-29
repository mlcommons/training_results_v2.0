# Download and prepare the data

Please download and prepare the data as described [here](https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/README.md#download-and-prepare-the-data).

After preparation, you can see the directories are like:

```
<BASE_DATA_DIR>
                     |_ phase1                                   # checkpoint to start from tf1
                     |_ hdf5  
                           |_ eval                               # evaluation chunks in binary hdf5 format fixed length 
                           |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length
                           |_ training_4320                      # 
                              |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length
                              |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length
```

# Build the docker image

We provide you two ways to build the docker image to run tests.

## Build the docker image with pre-built binaries

We built some necessary binaries beforehand, and you can build the docker image fast.

```
bash Dockerfiles/build_image_fast.sh
```

After the command finishes, you will get the docker image named `nvcr.io/nvidia/pytorch:22.04-py3-paddle-fast-test`.

## Build the docker image from scratch

This method would take a long time. It would contain the following steps:

- Build the docker image which can compile the PaddlePaddle source code.
- Compile the PaddlePaddle source code.
- Compile the PaddlePaddle external operators.
- Compile the PaddlePaddle external pybind functions.

You can run the steps above by using the following command.

```
bash Dockerfiles/build_image_from_scratch.sh
```

After the command finishes, you will get the docker image named `nvcr.io/nvidia/pytorch:22.04-py3-paddle-dev-test`.

# Prepare the checkpoint file

Originally, the checkpoint of the BERT model is generated from TensorFlow. We can convert the original TensorFlow checkpoint file to the Python dictionary like this and dump the dictionary using the Python pickle module.

```python
{
  "bert/encoder/layer_0/attention/self/query/kernel": numpy.ndarray(...),
  "bert/encoder/layer_0/attention/self/query/bias": numpy.ndarray(...),
  ...
}
```

In this way, we can run tests without installing TensorFlow again after conversion. You can use the following command to convert the original TensorFlow checkpoint file:

```python
python models/load_tf_checkpoint.py \
    <BASE_DATA_DIR>/phase1/model.ckpt-28252 \
    <BASE_DATA_DIR>/phase1/model.ckpt-28252.tf_pickled
```

# Run the tests

```
export NEXP=10 # the trial test number
export BASE_DATA_DIR=<your_bert_data_dir>
export CONT=<your_docker_image_name>
STAGE=run bash run_with_docker.sh
```