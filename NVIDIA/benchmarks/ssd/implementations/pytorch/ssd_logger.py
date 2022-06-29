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

import os
from functools import wraps

import utils
from mlperf_logging import mllog


class SSDLogger:
    def __init__(self, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(default_stack_offset=default_stack_offset,
                     filename=(filename or os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"),
                     root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))

    @property
    def rank(self):
        return utils.get_rank()

    def event(self, sync=False, sync_group=None, log_rank=None, *args, **kwargs):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            utils.barrier(group=sync_group)
        if log_rank:
            self.mllogger.event(*args, **kwargs)

    def start(self, sync=False, sync_group=None, log_rank=None, *args, **kwargs):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            utils.barrier(group=sync_group)
        if log_rank:
            self.mllogger.start(*args, **kwargs)

    def end(self, sync=False, sync_group=None, log_rank=None, *args, **kwargs):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            utils.barrier(group=sync_group)
        if log_rank:
            self.mllogger.end(*args, **kwargs)


mllogger = SSDLogger()
