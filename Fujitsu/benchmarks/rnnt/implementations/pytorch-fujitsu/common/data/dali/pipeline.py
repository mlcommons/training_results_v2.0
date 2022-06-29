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

from nvidia import dali
import torch
import numpy as np
import numpy as np


class PipelineParams:
    def __init__(
            self,
            sample_rate=16000,
            max_duration=float("inf"),
            normalize_transcripts=True,
            trim_silence=False,
            speed_perturbation=None
        ):
        pass

class SpeedPerturbationParams:
    def __init__(
            self,
            min_rate=0.85,
            max_rate=1.15,
            p=1.0,
        ):
        pass


SILENCE_THRESHOLD=-60
PREEMPH_COEFF=0.97

def _dali_init_log(args: dict):
    if (not torch.distributed.is_initialized() or (
            torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):  # print once
        max_len = max([len(ii) for ii in args.keys()])
        fmt_string = '\t%' + str(max_len) + 's : %s'
        print('Initializing DALI with parameters:')
        for keyPair in sorted(args.items()):
            print(fmt_string % keyPair)


@dali.pipeline_def
def librispeech(
        gpu_pipeline,
        file_root,
        sample_rate,
        resample_range,
        nfft,
        spect_wind_len,
        spect_wind_step,
        nfilter,
        dither,
        sampler,
        synthetic_seq_len,
        in_mem_file_list,
        dont_use_mmap,
    ):
    _dali_init_log(locals())

    # distributed sampler creates separate file list for each worker,
    # so there is one shard and one shard ID
    shard_id = 0 if sampler.num_shards == 1 else sampler.rank

    if synthetic_seq_len is not None:
        files_arg = dict(file_list='/workspace/rnnt/rnnt_dali.file_list.synth')
    elif in_mem_file_list:
        files_arg = dict(files=sampler.files, labels=sampler.labels)
    else:
        files_arg = dict(file_list=sampler.get_file_list_path())

    audio, label = dali.fn.readers.file(
        **files_arg,
        file_root=file_root,
        name='Reader',
        pad_last_batch=True,
        shard_id=shard_id,
        num_shards=sampler.num_shards,
    )

    if sampler.pre_sort:
        epochs, iters, bs = sampler.pert_coeff.shape
        resample = dali.fn.external_source(
            source=iter(sampler.pert_coeff.view(-1, bs))
        )
    elif resample_range:
        resample = dali.fn.random.uniform(range=resample_range)
    else:
        resample = 1
    audio, _ = dali.fn.decoders.audio(audio, downmix=True, sample_rate=sample_rate*resample)

    begin, length = dali.fn.nonsilent_region(audio, cutoff_db=SILENCE_THRESHOLD)

    if gpu_pipeline:
        audio = audio.gpu()

    audio = dali.fn.slice(
        audio,
        begin,
        length,
        normalized_anchor=False,
        normalized_shape=False,
        axes=[0],
    )

    if synthetic_seq_len:
        audio = dali.fn.constant(fdata=100, shape=synthetic_seq_len[0], device=audio.device)

    distribution = dali.fn.random.normal(device=audio.device)
    audio = audio + distribution * dither

    audio = dali.fn.preemphasis_filter(audio)

    audio = dali.fn.spectrogram(
        audio,
        nfft=nfft,
        window_length=spect_wind_len,
        window_step=spect_wind_step,
    )

    audio = dali.fn.mel_filter_bank(
        audio,
        sample_rate=sample_rate,
        nfilter=nfilter,
    )

    audio = dali.fn.to_decibels(
        audio,
        multiplier=np.log(10),
        reference=1.0,
        cutoff_db=np.log(1e-20),
    )

    audio_len = dali.fn.shapes(audio)

    audio = dali.fn.normalize(audio, axes=[1])
    audio = dali.fn.pad(audio)

    return audio, label, audio_len


