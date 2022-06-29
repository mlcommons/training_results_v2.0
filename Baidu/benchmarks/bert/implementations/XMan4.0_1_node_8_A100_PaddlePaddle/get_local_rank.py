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

import os
import subprocess


def get_gpu_list():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        output = subprocess.check_output(
            ["sh", "-c", "nvidia-smi --list-gpus | wc -l"])
        gpu_num = int(output.strip())
        gpus = list(range(gpu_num))
    else:
        gpus = [int(g.strip()) for g in gpus.split(",") if g.strip()]
    return gpus


def get_local_rank():
    world_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    gpus = get_gpu_list()
    if world_rank is None:
        local_rank = gpus[0]
    else:
        local_rank = gpus[int(world_rank) % len(gpus)]
    return local_rank


if __name__ == "__main__":
    local_rank = get_local_rank()
    print(local_rank)
