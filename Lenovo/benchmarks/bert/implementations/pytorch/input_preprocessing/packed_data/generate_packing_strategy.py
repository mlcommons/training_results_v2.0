# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
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
import h5py
import numpy as np
from scipy import optimize
from functools import lru_cache, reduce
import argparse
import logging
from tqdm import tqdm
import glob
import random
import json
import time

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
    '--max_seq_length',
    type=int,
    default=512,
    help='maximum sequence length')
parser.add_argument(
    '--max_seq_per_sample',
    type=int,
    default=3,
    help='maximum sequence length')
parser.add_argument(
    '--shards_num',
    type=int,
    default=4320,
    help='number of output shards')
args = parser.parse_args()


@lru_cache(maxsize=None)
def packing_strategies(start, previous, target, depth):
    gap = target - start

    # The collection of possible strategies given the
    # starting sum, the target sum, and the available depth
    # strategy search is limited to increments greater or equal to previous
    strategies = []
    # Complete the packing with exactly 1 number
    if depth == 1:
        if gap >= previous:
            strategies.append([gap])

    # Complete the sample in "depth" steps, recursively
    else:
        for new in range(previous, gap + 1):

            new_gap = target - start - new
            if new_gap == 0:
                strategies.append([new])
            else:
                options = packing_strategies(start + new, new, target, depth - 1)

                for option in options:
                    if len(option) > 0:
                        strategies.append([new] + option)
    return strategies


def get_packing_recipe(histogram, max_sequence_length, max_sequences_per_pack=3):
    # Histogram of sequence lengths
    sequence_lengths = np.array([ item  for idx,i in enumerate(histogram) for item in [idx+1]*i])
    print("Begin packing pass".center(80, "_"))
    print(f"Unpacked mean sequence length: {sequence_lengths.mean():3.2f}")

    # Make sure all strategies are recipes to pack to the correct sequence length
    strategy_set = packing_strategies(0, 1, max_sequence_length, max_sequences_per_pack)
    for strategy in strategy_set:
        assert(sum(strategy) == max_sequence_length)
    num_strategies = len(strategy_set)
    print(f"Found {num_strategies} unique packing strategies.")

    # Solve the packing equation A@mixture = histogram
    A = np.zeros((max_sequence_length, num_strategies), dtype=np.int32)
    for i in range(num_strategies):
        strategy = strategy_set[i]
        for seq_len in strategy:
            A[seq_len - 1, i] += 1

    # short sequences are inexpensive to add, so should have low residual weights
    # to exactly minimize padding use w0 = np.arange(1, max_sequence_length + 1)
    # in practice the difference is negligible, but this converges faster
    padding_cutoff = 8
    w0 = np.ones([max_sequence_length])
    # w0 = np.linspace(1, max_sequence_length+1, max_sequence_length)/max_sequence_length  # padding minimization weight
    w0[:padding_cutoff] = padding_cutoff / (2 * max_sequence_length)
    w0 = np.sqrt(w0)

    # Starting values for the padding and the mixture
    padding = np.zeros([max_sequence_length], dtype=np.int32)
    mixture = np.zeros([num_strategies], dtype=np.int32)
    b = histogram + padding

    # Pack sequences as best as possible, then increase padding accordingly and repeat
    for i in range(0, 20):
        print(f"\nIteration: {i}: sequences still to pack: ", b.sum())
        start = time.time()
        partial_mixture, rnorm = optimize.nnls(np.expand_dims(w0, -1) * A, w0 * b)
        print(f"Solving nnls took {time.time() - start:3.2f} seconds.")
        print(f"Residual norm:  {rnorm:3.5e}")

        # Update mixture (round the floating point solution to integers)
        partial_mixture = np.where(partial_mixture < 2, np.rint(partial_mixture), np.floor(partial_mixture))

        # If partial mixture is empty (due to rounding) we follow the gradient
        # this usually happens when the number of examples is small i.e. ~100
        if partial_mixture.max() == 0:
            grad = A.T @ (b * np.arange(1, max_sequence_length + 1))
            k = int(b.sum() // 2) + 1
            topk = np.argsort(-grad)[:k]
            partial_mixture[topk] += 1

        # Update mixture
        mixture = mixture + partial_mixture

        # Compute the residuals
        residual = b - A @ partial_mixture
        print(f"Max residual:   {abs(residual).max()}")
        print(f"Residual on first 8 categories: {np.around(residual[:8], 4)}")
        print(f"Residual on last 8 categories:  {np.around(residual[-8:], 4)}")

        # Add padding based on deficit (negative residual)
        partial_padding = np.where(residual < 0, -residual, 0)
        print(f"Added {(partial_padding*np.arange(1,max_sequence_length+1)).sum():3.2e} tokens of padding.")
        padding = padding + partial_padding

        # Update the rhs vector (remaining surplus sequences)
        b = histogram + padding - A @ mixture
        assert np.all(b >= 0), b

        # Done iterating
        if b.sum() < 100:
            break

    # Make sure there is no remainder
    unpacked_seqlen = np.arange(1, max_sequence_length + 1)[b > 0]
    # Update the mixture to also covered the unpacked sequences
    for l in unpacked_seqlen:
        # Get the depth 1 strategy
        strategy = sorted([l, max_sequence_length - l])
        strategy_index = strategy_set.index(strategy)
        mixture[strategy_index] += b[l-1]
    b = histogram - A @ mixture
    padding = np.where(b < 0, -b, 0)
    b = histogram + padding - A @ mixture
    assert b.sum() == 0

    # Analyze result
    print("Done solving for packing order".center(80, "_"))
    num_padding_tokens = (np.arange(1, max_sequence_length + 1) * padding).sum()
    num_padding_tokens_original = (max_sequence_length - sequence_lengths).sum()
    print(f"Number of sequences dropped:  {b.sum()}")
    print(f"Number of strategies utilized: {np.count_nonzero(mixture)}")
    new_number_of_samples = int(mixture.sum())
    compression = 1 - new_number_of_samples / len(sequence_lengths)
    print(f"New number of samples: {new_number_of_samples:3.2f}, original {len(sequence_lengths)}. A compression ratio of {compression:3.3f}")
    print(f"The expected speed-up from packing: {1/(1-compression):3.3f}")
    upper_bound = 1.0 / (1 - ((1 - sequence_lengths / max_sequence_length).mean()))
    print(f"Theoretical upper bound on speed-up: {upper_bound:3.3f}")
    avg_sequences_per_sample = ((A.sum(0) * mixture).sum() - padding.sum()) / new_number_of_samples
    print(f"Average sequences/sample {avg_sequences_per_sample:3.5f}")
    print(f"Added {num_padding_tokens:3.2e} padding tokens. Original dataset used {num_padding_tokens_original:3.2e} padding tokens")
    efficiency = (new_number_of_samples*max_sequence_length - num_padding_tokens)/(new_number_of_samples*max_sequence_length)
    print(f"Packing efficiency (fraction of real tokens): {efficiency:3.4f}")

    print(f"Top 8 strategies")
    topK = np.argsort(-mixture)[:8]
    for i in topK:
        print(f"Strategy {strategy_set[i]} which is used {int(mixture[i])} times")
    print("".center(80, "_"))

    # Figure out the slicing that each strategy should use
    slicing = np.zeros_like(A)
    slicing[:, 1:] = np.cumsum(A * mixture, axis=1)[:, :-1]
    slicing = slicing.T

    mixture = mixture.astype(np.int64)
    return strategy_set, mixture, padding, slicing

hdf5_compression_method = None
_counts = []
for ifile_idx in tqdm(range(args.max_seq_length)):
    h5_ifile = h5py.File(f'{args.input_hdf5}/train_sentences_{ifile_idx+1:03}.hdf5', 'r')
    _counts.append(h5_ifile['input_ids'].shape[0])
    h5_ifile.close()

strategy_set, mixture, padding, slicing = get_packing_recipe(_counts, args.max_seq_length, args.max_seq_per_sample)

per_seqlength_idxs = [list(range(n)) for n in _counts]
for seq_length, pad in enumerate(padding):
    per_seqlength_idxs[seq_length].extend([None]*int(pad))
    random.shuffle(per_seqlength_idxs[seq_length])

all_strategies = [strategy_instance for strategy_idx, strategy in enumerate(strategy_set) for strategy_instance in [strategy]*mixture[strategy_idx]]
random.shuffle(all_strategies)

seq_length_idx = [0]*args.max_seq_length
assignments = []
for strategy_instance in all_strategies:
    assignment = []
    for seq_length in strategy_instance :
        if per_seqlength_idxs[seq_length-1][seq_length_idx[seq_length-1]] != None:
            assignment += [(seq_length, per_seqlength_idxs[seq_length-1][seq_length_idx[seq_length-1]])]
        seq_length_idx[seq_length-1] += 1
    assignments.append(assignment)

new_samples_count = len(assignments)
shard_size = new_samples_count//args.shards_num
reminder = new_samples_count%args.shards_num
shards=[shard_size + 1*(i<reminder) for i in range(args.shards_num)]

logging.info(f"Splitting strategies among shards ...")
offset=0
for shard_idx, shard in tqdm(enumerate(shards), total=len(shards)):
    with open(f'{args.output_hdf5}/shard_list_{shard_idx:05}.lst','w') as f:
        f.write(json.dumps(assignments[offset:offset+shard]))
    # pickle.dump(assignments[offset:offset+shard], open(f'{args.output_hdf5}/shard_list_{shard_idx:05}.pkl','wb'))
    offset += shard