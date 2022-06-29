# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import multiprocessing
import numpy as np
import torch
import torch.distributed as dist
from common.helpers import print_once
from . import pipeline
from . import iterator


class DaliDataLoader:
    """
    DataLoader is the main entry point to the data preprocessing pipeline.
    To use, create an object and then just iterate over `data_iterator`.
    DataLoader will do the rest for you.
    Example:
        data_layer = DataLoader(DaliTrainPipeline, path, json, bs, ngpu)
        data_it = data_layer.data_iterator
        for data in data_it:
            print(data)  # Here's your preprocessed data

    Args:
        device_type: Which device to use for preprocessing. Choose: "cpu", "gpu"
        pipeline_type: Choose: "train", "val"
    """

    def __init__(self, gpu_id, dataset_path: str, shuffle, config_data: dict, config_features: dict,
                 tokenizer, batch_size: int, sampler, pipeline_type: str, seed, grad_accumulation_steps: int = 1,
                 num_threads=multiprocessing.cpu_count(),
                 tokenized_transcript=False,
                 device_type: str = "gpu", synthetic_seq_len=None, 
                 in_mem_file_list=True, enable_prefetch=False, preproc=None, min_seq_split_len=-1,
                 pre_sort=False, dont_use_mmap=False):

        self.enable_prefetch = enable_prefetch
        self.prefetch_stream = torch.cuda.Stream()
        self.synthetic_seq_len = synthetic_seq_len
        self.min_seq_split_len = min_seq_split_len
        self.preproc = preproc
        self.pivot_len_cpu = torch.tensor(0, dtype=torch.int, device='cpu').pin_memory()

        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.drop_last = (pipeline_type == 'train')
        self.device_type = device_type
        self.pipeline_type = self._parse_pipeline_type(pipeline_type)

        if in_mem_file_list:
            assert (len(sampler.files) > 0 and len(sampler.labels) > 0), "Please run sampler.sample() first"
        else:
            assert sampler.file_list_path is not None, "Please run sampler.sample() first"
        self.dataset_size = sampler.get_dataset_size()
        print_once(f"Dataset read by DALI. Number of samples: {self.dataset_size}")

        sample_rate = config_data['sample_rate']

        speed_perturbation = config_data['speed_perturbation']
        if speed_perturbation is None:
            resample_range = None
        else:
            resample_range = [speed_perturbation['min_rate'], speed_perturbation['max_rate']]

        librispeech_pl = pipeline.librispeech(
            gpu_pipeline=self.device_type == 'gpu',
            file_root=dataset_path,
            sample_rate=sample_rate,
            resample_range=resample_range,
            nfft=config_features['n_fft'],
            spect_wind_len=sample_rate * config_features['window_size'],
            spect_wind_step=sample_rate * config_features['window_stride'],
            nfilter=config_features['n_filt'],
            dither=config_features['dither'],
            sampler=sampler,
            synthetic_seq_len=synthetic_seq_len,
            in_mem_file_list=in_mem_file_list,
            dont_use_mmap=dont_use_mmap,
            batch_size=self.batch_size,
            num_threads=num_threads,
            device_id=gpu_id,
            seed=seed,
        )
        librispeech_pl.build()

        transcripts = sampler.transcripts
        if not tokenized_transcript:
            transcripts = [tokenizer.tokenize(t) for t in transcripts]

        self._dali_data_iterator = iterator.LibriSpeechIterator(
            librispeech_pl,
            self._shard_size(),
            transcripts,
            preproc,
            sampler,
        )

    @staticmethod
    def _parse_pipeline_type(pipeline_type):
        pipe = pipeline_type.lower()
        assert pipe in ("train", "val"), 'Invalid pipeline type ("train", "val").'
        return pipe

    def _shard_size(self):
        """
        Total number of samples handled by a single GPU in a single epoch.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.drop_last:
            divisor = world_size * self.batch_size * self.grad_accumulation_steps
            return self.dataset_size // divisor * divisor // world_size
        else:
            return int(math.ceil(self.dataset_size / world_size))

    def __len__(self):
        """
        Number of batches handled by each GPU.
        """
        if self.drop_last:
            assert self._shard_size() % self.batch_size == 0, f'{self._shard_size()} {self.batch_size}'

        return int(math.ceil(self._shard_size() / self.batch_size))

    def __iter__(self):
        return self

    def maybe_split(self, audio):
        if self.min_seq_split_len <= 0:
            return audio
        else:
            second_segment_len = audio.size(0) - self.pivot_len_cpu
            if second_segment_len >= self.min_seq_split_len:
                audio = [
                    audio[:self.pivot_len_cpu],
                    audio[self.pivot_len_cpu:, :self.split_batch_size],
                ]
            return audio

    def fetch_next(self):
        audio, audio_len, transcripts, transcripts_len = next(self._dali_data_iterator)

        if self.synthetic_seq_len != None:
            transcripts = torch.randint(
                transcripts.max(),
                (self.batch_size, self.synthetic_seq_len[1]),
                device=transcripts.device,
                dtype=transcripts.dtype,
            )
            transcripts_len = torch.ones_like(transcripts_len) * self.synthetic_seq_len[1]

        max_f_len = audio.size(0)
        if self.min_seq_split_len > 0:
            data = audio, audio_len, transcripts, transcripts_len
            data = self._prepare_seq_split(*data)
            audio, audio_len, transcripts, transcripts_len = data

        if self.enable_prefetch:
            # use async copy as we don't really want to sync here
            self.preproc.get_meta_data(
                max_f_len,
                audio_len,
                transcripts,
                transcripts_len,
                async_cp=True,
            )

        return audio, audio_len, transcripts, transcripts_len

    def _prepare_seq_split(self, audio, audio_shape, transcripts, transcripts_lengths):
        idx_sorted = torch.argsort(audio_shape, descending=True)
        audio_shape_sorted = audio_shape[idx_sorted]
        audio_sorted = audio[:, idx_sorted]
        transcripts_sorted = transcripts[idx_sorted]
        transcripts_lengths_sorted = transcripts_lengths[idx_sorted]
        batch_size = audio_shape.size(0)
        self.split_batch_size = batch_size // 2  # currently only split once
        stack_factor = self.preproc.enc_stack_time_factor
        # make sure the first segment is multiple of stack_factor so that stack time can be done easily
        pivot_len = torch.div(audio_shape_sorted[self.split_batch_size] + stack_factor-1,
                              stack_factor, rounding_mode='trunc') * stack_factor
        # copy pivot len asyncly for later use
        self.pivot_len_cpu.copy_(pivot_len.detach(), non_blocking=True)
        return audio_sorted, audio_shape_sorted, transcripts_sorted, transcripts_lengths_sorted

    def __next__(self):
        if not self.enable_prefetch:
            return self.fetch_next()
        else:
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)
            # make sure all async copies are committed
            self.prefetch_stream.synchronize()

            if self.prefetched_data is None:
                raise StopIteration
            else:
                self.preproc.copy_metadata()

                audio, audio_len, transcripts, transcripts_len = self.prefetched_data
                audio = self.maybe_split(audio)

                return audio, audio_len, transcripts, transcripts_len

    def prefetch(self):
        with torch.cuda.stream(self.prefetch_stream):
            try:
                self.prefetched_data = self.fetch_next()
            except StopIteration:
                self.prefetched_data = None
