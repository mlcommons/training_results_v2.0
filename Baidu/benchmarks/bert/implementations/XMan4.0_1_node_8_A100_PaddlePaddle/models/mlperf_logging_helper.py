# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from mlperf_logging import mllog

mllogger = mllog.get_mllogger()


def _paddle_bert_print(logger,
                       key,
                       val=None,
                       metadata=None,
                       stack_offset=3,
                       namespace="paddle_mlperf"):
    logger(
        key=key,
        value=val,
        metadata=metadata,
        stack_offset=stack_offset,
        namespace=namespace)


def paddle_bert_print_start(key, val=None, metadata=None):
    _paddle_bert_print(mllogger.start, key, val, metadata)


def paddle_bert_print_end(key, val=None, metadata=None):
    _paddle_bert_print(mllogger.end, key, val, metadata)


def paddle_bert_print_event(key, val=None, metadata=None):
    _paddle_bert_print(mllogger.event, key, val, metadata)
