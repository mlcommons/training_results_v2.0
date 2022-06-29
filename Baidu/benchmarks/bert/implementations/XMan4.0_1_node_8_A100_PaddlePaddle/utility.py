#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""Contains common utility functions."""

import distutils.util
import os
import paddle
import distutils
from init_env import get_context

context = get_context()


def get_mpi_comm():
    return context.trainer_comm


def get_num_trainers():
    return context.trainer_num


def get_trainer_id():
    return context.trainer_id


def str2bool(s):
    return True if distutils.util.strtobool(s) else False


def use_nv_input():
    return str2bool(os.environ.get("USE_NV_INPUT", "0"))


def get_place():
    dev_id = int(os.environ.get('FLAGS_selected_gpus', '0'))
    return paddle.CUDAPlace(dev_id)


def get_scope():
    scope = paddle.static.global_scope()
    return scope
