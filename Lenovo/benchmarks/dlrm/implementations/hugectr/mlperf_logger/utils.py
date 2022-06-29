# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

import os
from mlperf_logging.mllog import constants as mlperf_constants
from mlperf_logging import mllog

mllogger = mllog.get_mllogger()


def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)


def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)


def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)


def _log_print(logger, *args, **kwargs):
    logger(*args, **kwargs, stack_offset=3)


def mlperf_submission_log(benchmark, time_ms=None):

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

    log_event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_BENCHMARK,
        value=benchmark)

    log_event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_ORG,
        value='NVIDIA')

    log_event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_STATUS,
        value='onprem')

    log_event(
        time_ms=time_ms,
        key=mlperf_constants.SUBMISSION_PLATFORM,
        value='{}xSUBMISSION_PLATFORM_PLACEHOLDER'.format(num_nodes))
