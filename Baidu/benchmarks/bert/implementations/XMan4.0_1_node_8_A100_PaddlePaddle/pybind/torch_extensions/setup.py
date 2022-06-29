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

import torch
import os
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from setuptools import setup, find_packages

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

setup(
    name="torch_ex",
    version="0.1",
    description="PyTorch Extensions written by Baidu",
    ext_modules=[
        CUDAExtension(
            name="torch_ex",
            sources=['torch_ex.cc'],
            extra_compile_args={
                "cxx": [
                    "-O3", "-DVERSION_GE_1_1", "-DVERSION_GE_1_3",
                    "-DVERSION_GE_1_5", "-fPIC"
                ],
                "nvcc": [
                    "-O3", "-DVERSION_GE_1_1", "-DVERSION_GE_1_3",
                    "-DVERSION_GE_1_5", "-Xcompiler='-fPIC'"
                ],
            },
            extra_link_args=['-fPIC', '-lnccl_wrapper'])
    ],
    cmdclass={"build_ext": BuildExtension})
