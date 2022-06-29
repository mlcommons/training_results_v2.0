#!/bin/bash

: "${TF_INPUT_DIR:?tfrecord input dir not set}"
: "${HDF5_OUTPUT_DIR:?hdf5 output dir not set}"

mkdir -p ${HDF5_OUTPUT_DIR}

tf_file_list=$(ls ${TF_INPUT_DIR} | grep 'part')
echo -e "tfrecord file list:\n${tf_file_list}"

for tf_file in ${tf_file_list}
do
    hdf5_file=${tf_file}.hdf5
    echo "transform tfrecord file ${tf_file} to hdf5 file ${hdf5_file}"
    python trans_tf_hdf5.py  --tf_input_dir=${TF_INPUT_DIR}/${tf_file} --hdf5_output_dir=${HDF5_OUTPUT_DIR}/${hdf5_file}
done