# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import os
import subprocess

from mlperf_logging import mllog
from mlperf_logging.mllog import constants

mllogger = mllog.get_mllogger()


def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)


def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)


def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)


def _log_print(logger, *args, **kwargs):
    if kwargs.pop('sync', False):
        barrier()
    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 3
    if 'value' not in kwargs:
        kwargs['value'] = None

    if kwargs.pop('log_all_ranks', False):
        log = True
    else:
        log = (get_rank() == 0)

    if log:
        logger(*args, **kwargs)


def mlperf_submission_log(benchmark):

    num_nodes = os.environ.get('SLURM_NNODES', 1)

    mllog.config(filename=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f'{benchmark}.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False

    log_event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark, )

    log_event(key=constants.SUBMISSION_ORG, value='NVIDIA')

    log_event(key=constants.SUBMISSION_DIVISION, value='closed')

    log_event(key=constants.SUBMISSION_STATUS, value='onprem')

    log_event(
        key=constants.SUBMISSION_PLATFORM,
        value=f'{num_nodes}xSUBMISSION_PLATFORM_PLACEHOLDER')
