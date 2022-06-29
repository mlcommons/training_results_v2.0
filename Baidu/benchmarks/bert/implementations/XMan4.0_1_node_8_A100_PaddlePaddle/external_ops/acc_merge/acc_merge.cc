// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"

std::vector<std::vector<int64_t>> AccMergeInferShape(
    const std::vector<int64_t> &acc, const std::vector<int64_t> &total) {
  return {{2}, {2}};
}

std::vector<paddle::DataType> AccMergeInferDType(paddle::DataType acc,
                                                 paddle::DataType total) {
  return {paddle::DataType::FLOAT64, paddle::DataType::INT64};
}

PD_BUILD_OP(acc_merge)
    .Inputs({"Acc", "Total"})
    .Outputs({"Out", "Step"})
    .SetInferShapeFn(PD_INFER_SHAPE(AccMergeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AccMergeInferDType));
