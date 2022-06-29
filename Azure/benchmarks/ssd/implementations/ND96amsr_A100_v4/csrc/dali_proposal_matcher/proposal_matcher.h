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

#ifndef DALI_PROPOSAL_MATCHER_H_
#define DALI_PROPOSAL_MATCHER_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace other_ns {

template <typename Backend>
class proposal_matcher : public ::dali::Operator<Backend> {
 public:
  inline explicit proposal_matcher(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {

    int gt = 1000;
    int preds = 120087;
    int num_chunks = (preds + 2047) / 2048;

    cudaMalloc(&d_best_pred_per_gt, gt * sizeof(float));
    cudaMalloc(&d_intergt, gt * num_chunks * sizeof(float));
    cudaMalloc(&d_pred_forgiven, preds * sizeof(unsigned char));
  }

  virtual inline ~proposal_matcher() {
    cudaFree(d_best_pred_per_gt);
    cudaFree(d_intergt);
    cudaFree(d_pred_forgiven);
  }

  proposal_matcher(const proposal_matcher&) = delete;
  proposal_matcher& operator=(const proposal_matcher&) = delete;
  proposal_matcher(proposal_matcher&&) = delete;
  proposal_matcher& operator=(proposal_matcher&&) = delete;

 protected:

  float *d_best_pred_per_gt, *d_intergt;
  unsigned char *d_pred_forgiven;

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    auto shape = input.shape();
    
    output_desc.resize(1);

    const int N = input.num_samples();
    output_desc[0].shape = shape;
    for (int i = 0; i < N; i++) {
       output_desc[0].shape.tensor_shape_span(i)[0] = 1;
       output_desc[0].shape.tensor_shape_span(i)[1] = shape[i][1];
    }

    //output_desc[0].type = input.type();
    output_desc[0].type = dali::DALI_INT64;

    return true;
  }

  void RunImpl(::dali::workspace_t<Backend> &ws) override;
};

}  // namespace other_ns

#endif  // DALI_PROPOSAL_MATCHER_H_

