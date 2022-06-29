from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import argparse
import logging
from tqdm import tqdm
from itertools import repeat
import json
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
    '--assignment_file',
    type=str,
    default='pkl',
    help='assignment file')
parser.add_argument(
    '--output_hdf5',
    type=str,
    default='',
    help='Output hdf5_dir path')
parser.add_argument(
    '--max_seq_length',
    type=int,
    default=512,
    help='maximum sequence length')
args = parser.parse_args()


ifile_handles={}
for ifile_idx in tqdm(range(args.max_seq_length)):
    handle = h5py.File(f'{args.input_hdf5}/train_sentences_{ifile_idx+1:03d}.hdf5', 'r')
    ifile_handles[ifile_idx] = [
        handle['input_ids'][:],
        handle['input_mask'][:],
        handle['segment_ids'][:],
        handle['masked_lm_positions'][:],
        handle['masked_lm_ids'][:],
        handle['next_sentence_labels'][:]
    ]
    handle.close()

def create_assignment(i):

    with open(f'{args.assignment_file}/shard_list_{i:05}.lst','r') as f:
        assignments = json.load(f)    
    # assignments = pickle.load(open(f'/output/shards/shard_list_{i:05}.pkl','rb'))
    n_samples_in_this_shard = len(assignments)
    hdf5_compression_method = None

    ofile_handle = h5py.File(f"output_{i:05}", "w")
    input_ids = ofile_handle.create_dataset('input_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    segment_ids = ofile_handle.create_dataset('segment_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)
    packed_input_len = ofile_handle.create_dataset('packed_input_len', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    masked_lm_positions = ofile_handle.create_dataset('masked_lm_positions', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    masked_lm_ids = ofile_handle.create_dataset('masked_lm_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    packed_masked_lm_len = ofile_handle.create_dataset('packed_masked_lm_len', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    next_sentence_labels = ofile_handle.create_dataset('next_sentence_labels', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)

    for oindex, assignment in tqdm(enumerate(assignments), total=n_samples_in_this_shard):
        _input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids, _next_sentence_labels, _packed_input_len, _packed_masked_lm_len = [],[],[],[],[],[],[]
        _masked_lm_positions_offset = 0
        for seq_length, iindex in assignment:

            valid_input = sum(ifile_handles[seq_length-1][1][iindex])
            valid_lm_positions = sum(ifile_handles[seq_length-1][3][iindex] != 0)
            _input_ids += list(ifile_handles[seq_length-1][0][iindex, :valid_input])
            _segment_ids += list(ifile_handles[seq_length-1][2][iindex, :valid_input])
            _masked_lm_positions += list(ifile_handles[seq_length-1][3][iindex, :valid_lm_positions] + _masked_lm_positions_offset)
            _masked_lm_ids += list(ifile_handles[seq_length-1][4][iindex, :valid_lm_positions])
            _next_sentence_labels += [ifile_handles[seq_length-1][5][iindex]]
            _packed_input_len += [valid_input]
            _packed_masked_lm_len += [valid_lm_positions]
            _masked_lm_positions_offset += valid_input

        input_ids[oindex] = _input_ids
        segment_ids[oindex] = _segment_ids
        masked_lm_positions[oindex] = _masked_lm_positions
        masked_lm_ids[oindex] = _masked_lm_ids
        next_sentence_labels[oindex] = _next_sentence_labels
        packed_input_len[oindex] =  _packed_input_len
        packed_masked_lm_len[oindex] = _packed_masked_lm_len

    ofile_handle.flush()
    ofile_handle.close()
    logging.info(f"assignment {i}: {oindex+1} samples written.")

assignments = glob.glob(f'{args.assignment_file}/shard_list_*')
print(assignments)
with ProcessPoolExecutor(max_workers=32) as executor:    
    for partial_result in executor.map(create_assignment, list(range(len(assignments)))):
        pass
