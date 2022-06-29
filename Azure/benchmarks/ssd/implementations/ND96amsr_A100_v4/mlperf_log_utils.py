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

import collections
import os
import subprocess
from mlperf_logging import mllog
from mlperf_logging.mllog import constants as mlperf_constants


def mlperf_submission_log(benchmark):

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

    mllogger = mllog.get_mllogger()
    mllogger.event

    mllogger.event(key=mlperf_constants.SUBMISSION_BENCHMARK, value=benchmark)
    mllogger.event(key=mlperf_constants.SUBMISSION_ORG, value='NVIDIA')
    mllogger.event(key=mlperf_constants.SUBMISSION_DIVISION, value='closed')
    mllogger.event(key=mlperf_constants.SUBMISSION_STATUS, value='onprem')
    mllogger.event(key=mlperf_constants.SUBMISSION_PLATFORM, value=f'{num_nodes}xSUBMISSION_PLATFORM_PLACEHOLDER')
