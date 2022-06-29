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

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch


class LibriSpeechIterator:
    def __init__(self, pipe, shard_size, transcripts, preproc, sampler):
        self.preproc = preproc
        self.transcripts = list(transcripts.values())
        self.transcript_lens = torch.tensor([len(t) for t in self.transcripts], dtype=torch.int32)

        if not sampler.bucketing:
            self.iter = DALIGenericIterator(
                [pipe],
                ['audio', 'label', 'audio_len'],
                dynamic_shape=True,
                auto_reset=True,
                reader_name='Reader',
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
        else:
            self.iter = DALIGenericIterator(
                [pipe],
                ['audio', 'label', 'audio_len'],
                dynamic_shape=True,
                auto_reset=True,
                size=shard_size,
            )

    def __iter__(self):
        return self

    def __next__(self):
        d = next(self.iter)[0]
        audio, audio_len, label = d['audio'], d['audio_len'], d['label']

        # pipeline.PARTIAL outputs empty tensor on workers without samples
        # in the last iteration. Train code doesn't support that.
        # Raise StopIteration istead.
        if d['audio'].shape[0] == 0:
            return next(self.iter)

        audio_len = audio_len[:, 1]
        if self.preproc is not None:
            audio, audio_len = self.preproc.preproc_func(audio, audio_len)

        transcripts = [torch.tensor(self.transcripts[i]) for i in label]
        transcripts = torch.nn.utils.rnn.pad_sequence(
            transcripts,
            batch_first=True
        ).cuda()

        transcripts_lengths = torch.index_select(self.transcript_lens, 0, label.view(-1))
        return audio, audio_len, transcripts, transcripts_lengths.cuda()

