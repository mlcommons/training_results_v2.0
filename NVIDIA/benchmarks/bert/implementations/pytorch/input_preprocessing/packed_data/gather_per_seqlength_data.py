import h5py
import numpy as np
from collections import defaultdict
import itertools
import argparse
import logging
from tqdm import tqdm
import glob

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
    description="Training data sharding for BERT.")
parser.add_argument(
    '--input_hdf5',
    type=str,
    default='hdf5',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5',
    type=str,
    default='',
    help='Output hdf5_dir path')
parser.add_argument(
    '--seq_length',
    type=int,
    default=512,
    help='Output hdf5_dir path')
args = parser.parse_args()

hdf5_compression_method = None

seq_length = args.seq_length-1
logging.info("Reading input and filling per seq_length file ...")
offset = 0 
_input_ids=[]
_input_mask=[]
_segment_ids=[]
_masked_lm_positions=[]
_masked_lm_ids=[]
_next_sentence_labels=[]
_counts=[]

part_dirs = glob.glob(f'{args.input_hdf5}/part-*/')
for ifile_idx in tqdm(part_dirs, total=len(part_dirs)):
    h5_ifile = h5py.File('{}/seqlength_{:03d}.hdf5'.format(ifile_idx, seq_length+1), 'r')
    _counts.append(h5_ifile['input_ids'].shape[0])
    _input_ids.append(h5_ifile['input_ids'][:])
    _input_mask.append(h5_ifile['input_mask'][:])
    _segment_ids.append(h5_ifile['segment_ids'][:])
    _masked_lm_positions.append(h5_ifile['masked_lm_positions'][:])
    _masked_lm_ids.append(h5_ifile['masked_lm_ids'][:])
    _next_sentence_labels.append(h5_ifile['next_sentence_labels'][:])
    h5_ifile.close()

ofile_handle = h5py.File(f"{args.output_hdf5}/train_sentences_{seq_length+1:03}.hdf5", "w")
ofile_handle.create_dataset('input_ids', data=np.concatenate(_input_ids), dtype=np.dtype('int16'), compression=hdf5_compression_method)
ofile_handle.create_dataset('input_mask', data=np.concatenate(_input_mask), dtype=np.dtype('int8'), compression=hdf5_compression_method)
ofile_handle.create_dataset('segment_ids', data=np.concatenate(_segment_ids), dtype=np.dtype('int8'), compression=hdf5_compression_method)
ofile_handle.create_dataset('masked_lm_positions', data=np.concatenate(_masked_lm_positions), dtype=np.dtype('int16'), compression=hdf5_compression_method)
ofile_handle.create_dataset('masked_lm_ids', data=np.concatenate(_masked_lm_ids), dtype=np.dtype('int16'), compression=hdf5_compression_method)
ofile_handle.create_dataset('next_sentence_labels', data=np.concatenate(_next_sentence_labels), dtype='i1', compression=hdf5_compression_method)
ofile_handle.flush()
ofile_handle.close()
