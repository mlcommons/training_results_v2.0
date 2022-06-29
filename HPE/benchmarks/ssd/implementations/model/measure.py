# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import torch
import time
import numpy as np
import pickle

IS_PROFILE = False
IS_WALL_TIME = False


class stats_wrapper:
    def __init__(self):
        self.records = {}
        self.warmup_t = 0

    def print_all(self):
        print('>>> START STATS PRINT <<<')
        for k, v in self.records.items():
            samples = np.asarray(v['samples'])

            mean = np.mean(samples)
            standard_deviation = np.std(samples)
            distance_from_mean = abs(samples - mean)
            max_deviations = 2
            not_outlier = distance_from_mean < max_deviations * standard_deviation
            samples_ = samples[not_outlier]

            avg = samples_.mean()
            var = samples_.var()
            print('{}, {}, {}, {}, {}'.format(k, avg * 1000, var * 1000, samples.max() * 1000, samples.min() * 1000))
        print('>>> END STATS PRINT <<<')
        pickle.dump(self.records, open(b"records.pkl", "wb"))

    def create(self, k):
        if k not in self.records:
            self.records[k] = {'n': self.warmup_t * (-1), 'samples': []}

    def add(self, k, v):
        self.records[k]['samples'].append(v)
        self.records[k]['n'] += 1


class measure_t:
    def __init__(self, name, enable=True):
        self.name = name
        self.t0, self.t1 = 0, 0

        self.enable = enable
        self.is_running = False

        if enable:
            stats.create(self.name)

    def __enter__(self):
        if not self.enable:
            return

        self.start()

    def __exit__(self, type, value, traceback):
        if not self.enable:
            return

        self.stop()

    def start(self):
        if not self.enable:
            return

        if IS_PROFILE:
            torch.cuda.nvtx.range_push(self.name)

        if IS_WALL_TIME:
            torch.cuda.synchronize()
            self.t0 = time.time()

        self.is_running = True

    def stop(self):
        if not self.enable:
            return

        if self.is_running:
            if IS_PROFILE:
                torch.cuda.nvtx.range_pop()

            if IS_WALL_TIME:
                torch.cuda.synchronize()
                self.t1 = time.time()
                delta = self.t1 - self.t0
                stats.add(self.name, delta)

            self.is_running = False


stats = stats_wrapper()
