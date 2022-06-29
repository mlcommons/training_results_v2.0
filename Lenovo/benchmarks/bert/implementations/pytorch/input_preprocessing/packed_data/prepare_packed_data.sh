#!/bin/bash
# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

function usage()
{
   cat << HEREDOC

   Usage: $progname [-o|--outputdir PATH] [-h|--help TIME_STR]

   optional arguments:
     -h, --help            show this help message and exit
     -o, --outputdir PATH  pass in a localization of resulting dataset
     -s, --skip-download   skip downloading raw files from GDrive (assuming it already has been done)
     -p, --shards          number of resulting shards. For small scales (less than 256 nodes) use 2048. For sacles >256 4320 is recommended (default 4320)

HEREDOC
}

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

#if no arguments passed
DATADIR=/workspace/bert_data
SKIP=0
SHARDS=4320

#parse passed arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--help)
      usage
      exit 0
      ;;
    -o|--outputdir)
      DATADIR="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--shards)
      SHARDS="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--skip-download)
      SKIP=1
      shift
      ;;
    *)    # unknown option
      usage
      exit 1
      ;;
  esac
done


echo "Preparing Mlperf BERT dataset in ${DATADIR}"
mkdir -p ${DATADIR}

if (( SKIP==0 )) ; then
    
    mkdir -p ${DATADIR}/phase1 && cd ${DATADIR}/phase1
    ### Download 
    # bert_config.json
    gdown https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW    
    # vocab.txt
    gdown https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1
    
    ### Download dataset
    mkdir -p ${DATADIR}/download && cd ${DATADIR}/download
    # md5 sums
    gdown https://drive.google.com/uc?id=1tmMgLwoBvbEJEHXh77sqrXYw5RpqT8R_
    # processed chunks
    gdown https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_
    # unpack results and verify md5sums
    tar -xzf results_text.tar.gz && (cd results4 && md5sum --check ../bert_reference_results_text_md5.txt)
    
    
    ### Download TF1 checkpoint
    mkdir -p ${DATADIR}/phase1 && cd ${DATADIR}/phase1
    # model.ckpt-28252.data-00000-of-00001
    gdown https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_
    # model.ckpt-28252.index
    gdown https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0
    # model.ckpt-28252.meta
    gdown https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv
    
    cd ${DATADIR}
    
fi


mkdir -p ${DATADIR}/per_seqlen_parts
for shard in `seq -w 00000 00499`; do
    mkdir -p ${DATADIR}/per_seqlen_parts/part-${shard}
done

# Parallelize over $CPUS cores
CPUS=64
seq -w 00000 00499 | xargs --max-args=1 --max-procs=$CPUS -I{} python create_per_seqlength_data.py \
    --input_file ${DATADIR}/download/results4/part-{}-of-00500 \
    --output_file ${DATADIR}/per_seqlen_parts/part-{} \
    --vocab_file ${DATADIR}/phase1/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=10

#Merge all results
mkdir -p ${DATADIR}/per_seqlen
seq 0 511 | xargs --max-args=1 --max-procs=$CPUS -I{} python ./gather_per_seqlength_data.py \
    --input_hdf5 /workspace/bert_data/per_seqlen_parts \
    --output_hdf5 /workspace/bert_data/per_seqlen \
    --seq_length {}

#Generate sub-optimal packing strategy based on lenghts distribution of training set and store samples-based lists per shard
python ./generate_packing_strategy.py \
    --input_hdf5 /workspace/bert_data/per_seqlen \
    --output_hdf5 /workspace/bert_data/packed_data \
    --max_seq_length 512 \
    --max_seq_per_sample 3 \
    --shards_num ${SHARDS} 

# Create training set shards based on generated lists
python create_packed_trainset.py \
    --input_hdf5 ${DATADIR}/per_seqlen \
    --assignment_file ${DATADIR}/packed_data \
    --output_hdf5 ${DATADIR}/packed_data
