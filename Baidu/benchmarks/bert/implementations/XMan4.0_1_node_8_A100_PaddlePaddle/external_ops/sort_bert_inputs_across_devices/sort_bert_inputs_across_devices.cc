// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#define CHECK_GPU_INPUT(__x)                                 \
  do {                                                       \
    if (__x.place().GetType() != phi::AllocationType::GPU) { \
      PD_THROW(#__x " must be GPU Tensor.");                 \
    }                                                        \
  } while (0)

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> GPUSortBERTInputsAcrossDevices(
    const paddle::Tensor &input_ids,
    const paddle::Tensor &segment_ids,
    const paddle::Tensor &input_mask,
    const paddle::Tensor &masked_lm_labels,
    const paddle::Tensor &next_sentence_labels,
    int max_batch_size,
    int ring_id,
    int device_id,
    int num_devices);
#endif

static std::vector<paddle::Tensor> SortBERTInputsAcrossDevices(
    const paddle::Tensor &input_ids,
    const paddle::Tensor &segment_ids,
    const paddle::Tensor &input_mask,
    const paddle::Tensor &masked_lm_labels,
    const paddle::Tensor &next_sentence_labels,
    int max_batch_size,
    int ring_id,
    int device_id,
    int num_devices) {
#ifdef PADDLE_WITH_CUDA
  CHECK_GPU_INPUT(input_ids);
  CHECK_GPU_INPUT(segment_ids);
  CHECK_GPU_INPUT(input_mask);
  CHECK_GPU_INPUT(masked_lm_labels);
  CHECK_GPU_INPUT(next_sentence_labels);
  return GPUSortBERTInputsAcrossDevices(input_ids,
                                        segment_ids,
                                        input_mask,
                                        masked_lm_labels,
                                        next_sentence_labels,
                                        max_batch_size,
                                        ring_id,
                                        device_id,
                                        num_devices);
#else
  PADDLE_THROW(platform::errors::InvalidArgument("Does not support CPU yet."));
#endif
}

static std::vector<std::vector<int64_t>> SortBERTInputsAcrossDevicesInferShape(
    const std::vector<int64_t> &input_ids_shape,
    const std::vector<int64_t> &segment_ids_shape,
    const std::vector<int64_t> &input_mask_shape,
    const std::vector<int64_t> &masked_lm_labels_shape,
    const std::vector<int64_t> &next_sentence_labels_shape,
    const int &max_batch_size,
    const int &ring_id,
    const int &device_id,
    const int &num_devices) {
  return {input_ids_shape,
          segment_ids_shape,
          input_mask_shape,
          masked_lm_labels_shape,
          next_sentence_labels_shape};
}

static std::vector<paddle::DataType> SortBERTInputsAcrossDevicesInferDType(
    paddle::DataType input_ids_dtype,
    paddle::DataType segment_ids_dtype,
    paddle::DataType input_mask_dtype,
    paddle::DataType masked_lm_labels_dtype,
    paddle::DataType next_sentence_labels_dtype) {
  return {input_ids_dtype,
          segment_ids_dtype,
          input_mask_dtype,
          masked_lm_labels_dtype,
          next_sentence_labels_dtype};
}

PD_BUILD_OP(sort_bert_inputs_across_devices)
    .Inputs({"InputIds",
             "SegmentIds",
             "InputMask",
             "MaskedLMLabels",
             "NextSentenceLabels"})
    .Outputs({"InputIdsOut",
              "SegmentIdsOut",
              "InputMaskOut",
              "MaskedLMLabelsOut",
              "NextSentenceLabelsOut"})
    .Attrs({"max_batch_size: int",
            "ring_id: int",
            "device_id: int",
            "num_devices: int"})
    .SetKernelFn(PD_KERNEL(SortBERTInputsAcrossDevices))
    .SetInferShapeFn(PD_INFER_SHAPE(SortBERTInputsAcrossDevicesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SortBERTInputsAcrossDevicesInferDType));
