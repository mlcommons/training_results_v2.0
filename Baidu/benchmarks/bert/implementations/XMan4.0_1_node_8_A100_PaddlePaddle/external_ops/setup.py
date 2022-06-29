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
import paddle.fluid.core as core
from paddle.utils.cpp_extension import CUDAExtension, setup

compile_dir = os.environ.get('COMPILE_DIR', '/limin29/Paddle/build')
apex_dir = os.environ.get('APEX_DIR', '/limin29/apex/build_scripts')
apex_lib_dir = '/usr/local/lib'

define_macros = []
if core.is_compiled_with_mkldnn():
    define_macros.append(('PADDLE_WITH_MKLDNN', None))
if core.is_compiled_with_nccl():
    define_macros.append(('PADDLE_WITH_NCCL', None))

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=[
            './fused_dense_op/fused_dense_cuda.cc',
            './fused_dense_op/fused_dense_cuda.cu',
            './fused_dropout_residual_ln/fused_dropout_residual_ln_cuda.cc',
            './fused_dropout_residual_ln/fused_dropout_residual_ln_cuda.cu',
            './fmhalib/fmha_cuda.cc',
            './fmhalib/fmha_cuda.cu',
            './sort_bert_inputs_across_devices/sort_bert_inputs_across_devices.cc',
            './sort_bert_inputs_across_devices/sort_bert_inputs_across_devices.cu',
            './lr_op/lr_op_cuda.cc',
            './lr_op/lr_op_cuda.cu',
            './acc_merge/acc_merge.cc',
            './acc_merge/acc_merge.cu',
        ],
        extra_objects=[os.path.join(apex_lib_dir, 'libfmha.so')],
        include_dirs=[apex_dir],
        library_dirs=[apex_lib_dir],
        extra_link_args=['-lfmha', '-ldl', '-lcublas'],
        _compile_dir=compile_dir,
        define_macros=define_macros))
