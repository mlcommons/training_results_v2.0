# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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

import mxnet as mx
import mxnet.cuda_utils as cu
import graph_wrapper

a = mx.nd.zeros((10,10), ctx=mx.gpu())
mx.nd.waitall()
mx.nd._internal._plus_scalar(a, out=a, scalar=2)
mx.nd.waitall()
graph_wrapper.start_capture(0, 0, [a])
mx.nd._internal._plus_scalar(a, out=a, scalar=5)
graph_wrapper.end_capture(0, 0, [a])
mx.nd.waitall()
graph_wrapper.graph_replay(0, 0, [], [a])
graph_wrapper.graph_replay(0, 0, [], [a])
graph_wrapper.graph_replay(0, 0, [], [a])
graph_wrapper.graph_replay(0, 0, [], [a])
mx.nd.waitall()
print(a)

graph_wrapper.finalize()
