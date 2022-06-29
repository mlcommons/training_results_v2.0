# Download source code 
Download the impelementation codes optimized by Samsung with Samsung licensed. 
```shell
./download_code.sh
```

# Pre-trained checkpoint and bert config json file

1. Location of checkpoint and bert config json file

   This [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains these files.
   * TensorFlow checkpoint (tf1_ckpt) containing the pre-trained weights.
   * Config file (bert_config.json) which specifies the hyperparameters of the model.
2. Checkpoint conversion
```shell
python convert_tf_checkpoint.py --tf_checkpoint <path/to/checkpointdir_phase1/model.ckpt-28252.index> --bert_config_path <path/to/checkpointdir_phase1/bert_config.json> --output_checkpoint model.ckpt-28252.pt
```

# Our primay opimization

1. Complete usage of Pytorch DDP and ADAM optimizer for large batch training with communication/computation overlap

2. Bucket-wise gradient clipping before all-reduce that combines the advantages of clipping before all-reduce and clipping after all-reduce

3. Efficient load balancing of input data for increasing GPU utilization.

We recorded a time of 22.3 seconds on 1024 NVIDIA A100 GPUs and 21.4 seconds on 1368 NVIDIA A100 GPUs with the network bandwith 100 GB/s. 

# Download and preprocess datasets

1. Download dataset and generate the TFRecords for training data and eval data 

   [BERT Wikipedia dataset preparation](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets)
2. Convert training data and eval data from TFRecords to HDF5
   ```shell
   TF_INPUT_DIR=<path/to/tfrecord_input_dir> HDF5_OUTPUT_DIR=<path/to/hdf5_output_dir> ./run_trans_tfrecord_to_hdf5.sh
   ```
3. 4bins training data

   We split dataset to enable data-load balacning and it can reduce communication overhead. 

   Based on the sequence length distribution, split HDF5 training data into 4 part:
   
   part 1:  0 < sequence length <= 128
   
   part 2: 128 < sequence length <= 256
      
   part 3: 256 < sequence length <= 384
      
   part 4: 384 < sequence length <= 512
   
   The output_dir contains 4 sub-directories 128, 256, 384 and 512.

```shell
cd cleanup_scripts
python run_split_and_chop_hdf5_files.py --input_dir=<path/to/hdf5_datadir> --output_dir=<path/to/4bins_training_datadir>
```

# Prepare the environment

* Create a virtualenv and install the required packages:
```bash
virtualenv venv -p python3.8.7
source venv/bin/activate
pip install -r requirements.txt

# Install mlperf-logging Python package
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git reset --hard d06404fecab73f152c6cbb89ac2c2e9b7fc24124
git submodule update --init --recursive
git apply ../patch_for_mlperf_trining_v1.1_by_samsung.patch
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--distributed_adam" --global-option="--distributed_lamb" --global-option="--bnp" --global-option="--xentropy" --global-option="--fast_layer_norm" --global-option="--deprecated_fused_adam"  --global-option="--fmha"  --global-option="--fast_multihead_attn" ./

# Compile mhalib
cd mhalib
python setup.py build
cp build/lib*/mhalib* ../
```

* Other software requirements

| Softeware      |   Version      |
|----------------|  ------------: |
| python         |          3.8.7 |
| pytorch        |       1.10.0a0 |
| NCCL           |         2.11.4 |
| CUDA           |         11.4.2 |
| cudnn          |       8.2.4.15 |
| cublas         |      11.6.1.51 |
| nvidia driver  |     470.103.01 |
| mofed version  |      5.4-1.0.3 |

# Run the model

1. Set hosts address in run_multinode.sh
```bash 
export hosts=('192.168.16.1' '192.168.16.2')
```

2. Launch the training

   Use the following command to run the config_Samsung_Supercomputer21_DGXA100_128x8x16x1.sh in python virtual environment.
```shell
PYTHON=<path/to/python> DGXSYSTEM=Samsung_Supercomputer21_DGXA100_128x8x16x1 INPUT_DIR=<path/to/4bins_training_datadir> EVAL_DIR=<path/to/eval_datadir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1> NEXP=10 ./run_multinode.sh
```

# Appendix 

Our source code is based on MLPerf BERT v0.7, and all the files newly added and modified are as follows.

| File Name                  |   Status                       |  Description|
|----------------|  ------------  | -------------  |
| config_Samsung_Supercomputer21_DGXA100_128x8x16x1.sh       |       Newly added |The file contains configurations used for 1024 GPUs experiment.|
| config_Samsung_Supercomputer21_DGXA100_171x8x12x1.sh       |       Newly added |The file contains configurations used for 1368 GPUs experiment.|
| run_split_and_chop_hdf5_files.py           |          Newly added |The file is used for generating 4-bin training data.|
| mhalib/setup.py         |       Modified |The file is modified since CUDA upgraded.|
|optim/__init__.py         |        Newly added|The file is used as the entrance of "optim" module.|
| optim/acclip.py  |     Newly added |The file implements ACClip optimizer for trial.|
| optim/madgrad.py |    Newly added|The file implements MADGRAD optimizer for trial.|
|bind_launch.py|Newly added|The file is added for BERT training on python environment.|
|bind_pyt.py|Modified|The file is modified for the following items. <br/> (1) Log compliance; <br/> (2) Add new NUMA binding.|
|fmha.py|Newly added|The file is used for adding FMHA operator (refer to MLPerf v1.0).|
|mlperf_logger.py|Modified|The file is modified for log compliance.|
|modeling.py|Modified|The file is modified for adding FMHA (refer to MLPerf v1.0).|
|padding.py|Modified|The file is modified for adding FMHA (refer to MLPerf v1.0).|
|README.md|Modified|It is modified to run Samsung optimized implematation.|
|requirements.txt|Modified|The file shows required software version.|
|run_multinode.sh|Newly added|The file is startup script about how to run BERT training on python environment|
|run_pretraining.py|Modified|The file is modified for the following items. <br/>(1) Load splitting training data; <br/>(2) Add exchange padding feature (refer to MLPerf v1.0); <br/>(3) Add NCCL warmup (refer to MLPerf v1.0); <br/>(4) Add SAIT local/group exchange padding; <br/>(5) Add NCCL warmup for group exchange padding; <br/>(6) Add per-device local gradient clipping before all-reduce; <br/>(7) Add pytorch DDP.|
|schedulers.py|Modified|The file is modified for optimizing learning rate scheduler|
|utils.py |Modified|The file is modified for the following items. <br/>(1) Add get_optimzer() interface; <br/>(2) Add a batch sampler (SplitRandomSampler) for 4-bin splitting training data.|
