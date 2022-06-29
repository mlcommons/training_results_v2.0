// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//           http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_BOX_IOU_H_
#define DALI_BOX_IOU_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace other_ns {

template <typename Backend>
class box_iou : public ::dali::Operator<Backend> {
 public:
  inline explicit box_iou(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {}

  virtual inline ~box_iou() = default;

  box_iou(const box_iou&) = delete;
  box_iou& operator=(const box_iou&) = delete;
  box_iou(box_iou&&) = delete;
  box_iou& operator=(box_iou&&) = delete;

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<Backend> &ws) override {
    const auto &box1 = ws.template Input<Backend>(0);
    const auto &box2 = ws.template Input<Backend>(1);
    auto box1_shape = box1.shape();
    auto box2_shape = box2.shape();
    
    output_desc.resize(1);

    const int N = box1.num_samples();
    output_desc[0].shape = box1_shape;
    for (int i = 0; i < N; i++) {
       output_desc[0].shape.tensor_shape_span(i).back() = box2_shape[i][0];
    }

    output_desc[0].type = box2.type();
    return true;
  }

  void RunImpl(::dali::workspace_t<Backend> &ws) override;
};

}  // namespace other_ns

#endif  // DALI_BOX_IOU_H_

