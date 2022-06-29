# Download and prepare the data

Building the Docker container
```shell
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

Start the container interactively, mounting the directory you want to store the expieriment data as `/workspace/bert_data`
```
docker run -it --runtime=nvidia --ipc=host (...) -v /data/mlperf/bert:/workspace/bert_data mlperf-nvidia:language_model
```

Within the container, run
```
cd /workspace/bert
./input_preprocessing/prepare_data.sh --outputdir /workspace/bert_data
```
This script will download the required data and model files from [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) creating the following foldes structure
```
/workspace/bert_data/
                     |_ download
                        |_results4                               # 500 chunks with text data 
                        |_bert_reference_results_text_md5.txt    # md5 checksums for text chunks
                     |_ phase1                                   # checkpoint to start from (both tf1 and pytorch converted)
                     |_hdf5  
                           |_ eval                               # evaluation chunks in binary hdf5 format fixed length (not used in training, can delete after data   preparation)  
                           |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length *used for training*
                           |_ training                           # 500 chunks in binary hdf5 format 
                           |_ training_4320                      # 
                              |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length (not used in training, can delete after data   preparation)
                              |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length *used for training*
```   

The resulting HDF5 files store the training/evaluation dataset as a variable-length types (https://docs.h5py.org/en/stable/special.html). Note that these do not have a direct Numpy equivalents and require "by-sample" processing approach. The advantage is significant storage requirements reduction.

The prepare_data.sh script does the following:
* downloads raw data from GoogleDrive
* converts the training data to hdf5 format (each of 500 data chunks)
* splits the data into appropriate number of shards (for large scale training we recommend using 4320 shards - the default)
* 'compresses' the shards converting fixed-length hdf5 to variable-length hdf5 format
* applies the same procedure to evaluation data
* converts the seed checkpoint from tensorflow 1 to pytorch format

To verify correctness of resulting data files one may compute checksums for each of shards (using hdf5_md5.py script) and compere it with checksums in 4320_shards_varlength.chk or 2048_shards+varlength.chk files. Example of how to compute the checksums 

```bash
### Generate checksums to verify correctness of the process possibly paralellized with e.g. xargs and then sorted
for i in `seq -w 0000 04319`; do 
  python input_preprocessing/hdf5_md5.py \
    --input_hdf5 path/to/varlength/shards/part_${i}_of_04320.hdf5 
done | tee 4320_shards_varlength.chk
```

## Clean up

To de-clutter `bert_data/` directory, you can remove _download_, _training_, and _hdf5_4320_shards_uncompressed_ directories, but if disk space is not a concern, it might be good to keep these to debug any data preprocessing issue.

# Running the model

To run this model, use the following command. Replace the configuration script to match the system being used.
The experiment parameters like learning rate, maximum number of steps etc. are set in the config file named `config_{nodes}x{gpus per node}x{local batch size}x{gradien accumulation}.sh`, while the general system params like number of cores, sockets etc. are set in `config_DGXA100_common.sh`

```shell
source ./config_*.sh
sbatch -N${DGXNNODES} --ntasks-per-node=${DGXNGPU} --time=${WALLTIME} run.sub
```

## Alternative launch with nvidia-docker

```bash
source ./config_DGXA100_1x8x56x1.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/4320_shards_varlength/dir> DATADIR_PHASE2=<path/to/4320_shards_varlength/dir> EVALDIR=<path/to/eval_varlength/dir> CHECKPOINTDIR=<path/to/result/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/pytorch/ckpt/dir> ./run_with_docker.sh
```

You can also specify the data paths directly in `config_DGXA100_common.sh`.

## Multinode
For multi-node training, we use Slurm for scheduling and Pyxis to run our container.

## Configuration File Naming Convention

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>x<GRADIENT_ACCUMULATION_STEPS>.sh`.

### Example 1
A DGX1 system with 1 node, 8 GPUs per node, batch size of 6 per GPU, and 6 gradient accumulation steps would use `config_DGX1_1x8x6x6.sh`.

### Example 2
A DGX A100 system with 32 nodes, 8 GPUs per node, batch size of 20 per GPU, and no gradient accumulation would use `config_DGXA100_32x8x20x1.sh`


# Description of how the `results_text.tar.gz` file was prepared

1. First download the [wikipedia
   dump](https://drive.google.com/file/d/18K1rrNJ_0lSR9bsLaoP3PkQeSFO-9LE7/view?usp=sharing)
   and extract the pages The wikipedia dump can be downloaded from [this google
   drive](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT),
   and should contain `enwiki-20200101-pages-articles-multistream.xml.bz2` as
   well as the md5sum.

2. Run [WikiExtractor.py](https://github.com/attardi/wikiextractor), version
   e4abb4cb from March 29, 2020, to extract the wiki pages from the XML The
   generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for
   example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has
   100 files from wiki_00 to wiki_99, except the last sub directory. For the
   20200101 dump, the last file is FE/wiki_17.

3. Clean up and dataset seperation.  The clean up scripts (some references
   here) are in the scripts directory.  The following command will run the
   clean up steps, and put the resulted trainingg and eval data in ./results
   ./process_wiki.sh 'text/*/wiki_??'

4. After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 500 files named part-00xxx-of-00500 in the ./results directory, together with eval.md5 and eval.txt.

5. Exact steps (starting in the bert path)

```shell
cd input_preprocessing
mkdir -p wiki
cd wiki
# download enwiki-20200101-pages-articles-multistream.xml.bz2 from Google drive and check md5sum
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2
cd ..    # back to bert/input_preprocessing
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
git checkout e4abb4cbd
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/input_preprocessing/text
./process_wiki.sh './text/*/wiki_??'
```

MD5sums:

| File                                               |   Size (bytes) | MD5                              |
|----------------------------------------------------|  ------------: |----------------------------------|
| bert_config.json                                   |            314 | 7f59165e21b7d566db610ff6756c926b |
| vocab.txt                                          |        231,508 | 64800d5d8528ce344256daf115d4965e |
| model.ckpt-28252.index (tf1)                       |         17,371 | f97de3ae180eb8d479555c939d50d048 |
| model.ckpt-28252.meta (tf1)                        |     24,740,228 | dbd16c731e8a8113bc08eeed0326b8e7 |
| model.ckpt-28252.data-00000-of-00001 (tf1)         |  4,034,713,312 | 50797acd537880bfb5a7ade80d976129 |
| model.ckpt-28252.index (tf2)                       |          6,420 | fc34dd7a54afc07f2d8e9d64471dc672 |
| model.ckpt-28252.data-00000-of-00001 (tf2)         |  1,344,982,997 | 77d642b721cf590c740c762c7f476e04 |
| enwiki-20200101-pages-articles-multistream.xml.bz2 | 17,751,214,669 | 00d47075e0f583fb7c0791fac1c57cb3 |
| enwiki-20200101-pages-articles-multistream.xml     | 75,163,254,305 | 1021bd606cba24ffc4b93239f5a09c02 |

# Acknowledgements

We'd like to thank members of the ONNX Runtime team at Microsoft for their suggested performance optimization to reduce the size of the last linear layer to only output the fraction of tokens that participate in the MLM loss calculation.
