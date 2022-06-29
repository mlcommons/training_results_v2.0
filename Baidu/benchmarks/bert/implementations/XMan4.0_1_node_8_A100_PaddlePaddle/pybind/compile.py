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

import os
import paddle
import sys
import paddle.fluid.core as core
from paddle.utils.cpp_extension.extension_utils import _get_include_dirs_when_compiling

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

CMAKELISTS_TEMPLATE = '''
cmake_minimum_required(VERSION 3.4...3.18)
project(functions LANGUAGES CXX)

add_subdirectory(pybind11)

%s
%s

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC")

set(extension_name "functions")
add_definitions("-DMLPERF_EXTENSION_NAME=${extension_name}")
pybind11_add_module(${extension_name} functions.cc)
target_link_libraries(${extension_name} PRIVATE %s)
'''

compile_dir = os.environ["COMPILE_DIR"]
dir_lists = _get_include_dirs_when_compiling(compile_dir)
dirs = ["include_directories({})".format(d) for d in dir_lists]

macros = {}
if core.is_compiled_with_cuda():
    macros['PADDLE_WITH_CUDA'] = None
    macros['EIGEN_USE_GPU'] = None
if core.is_compiled_with_mkldnn():
    macros['PADDLE_WITH_MKLDNN'] = None
if core.is_compiled_with_nccl():
    macros['PADDLE_WITH_NCCL'] = None

macros = "\n".join([
    'add_definitions(-D{}{})'.format(k, '=' + str(v) if v is not None else "")
    for k, v in macros.items()
])

paddle_so = os.path.join(os.path.dirname(paddle.__file__), "fluid/core_avx.so")
cmakelists_context = CMAKELISTS_TEMPLATE % ("\n".join(dirs), macros, paddle_so)

with open("CMakeLists.txt", "w") as f:
    f.write(cmakelists_context)

cmake_args = {
    'CMAKE_BUILD_TYPE': 'Release',
    'PYBIND11_PYTHON_VERSION':
    '{}.{}'.format(sys.version_info.major, sys.version_info.minor),
}

cmd = "rm -rf pybind11 && ln -s {}/third_party/pybind/src/extern_pybind pybind11".format(
    compile_dir)
assert os.system(cmd) == 0

cmd = "rm -rf build && mkdir -p build && cd build && cmake .. {} && make -j `nproc`"
cmd = cmd.format(" ".join(
    ["-D{}={}".format(k, v) for k, v in cmake_args.items()]))
assert os.system(cmd) == 0
print(cmd)

so_file = [f for f in os.listdir('build') if f.endswith('.so')]
assert len(so_file) == 1
so_file = so_file[0]
assert os.system("cp build/{} .".format(so_file)) == 0
