// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "paddle/extension.h"

// Void paddle kernel funtion 
std::vector<paddle::Tensor> Kernel_Function(){
  return {};
}
std::vector<paddle::Tensor> Kernel_Function_Grad(){
  return {};
}

// AttentionMask
std::vector<std::vector<int64_t>> InferShape_AttentionMask(
    std::vector<int64_t> x_shape, std::vector<int64_t> y_shape) {
  return {{x_shape[0], 1, x_shape[1], x_shape[1]}};
}
std::vector<paddle::DataType> InferDtype_AttentionMask(paddle::DataType x_dtype, paddle::DataType y_dtype) {
  return {y_dtype};
}
PD_BUILD_OP(custom_AttentionMask)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Attrs({"dataType: std::string"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_AttentionMask))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_AttentionMask));

PD_BUILD_GRAD_OP(custom_AttentionMask)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("Y")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));


// Detach
PD_BUILD_OP(custom_Detach)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"pass_through_creation: int"})
    .SetKernelFn(PD_KERNEL(Kernel_Function));

PD_BUILD_GRAD_OP(custom_Detach)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));

// DropoutWithTrainingSwitch
std::vector<std::vector<int64_t>> InferShape_DropoutWithTrainingSwitch(
    std::vector<int64_t> x_shape, std::vector<int64_t> y_shape) {
  return {x_shape};
}
std::vector<paddle::DataType> InferDtype_DropoutWithTrainingSwitch(paddle::DataType x_dtype, paddle::DataType y_dtype) {
  return {x_dtype};
}
// mode is UINT32
PD_BUILD_OP(custom_DropoutWithTrainingSwitch)
    .Inputs({"X", "mode"})
    .Outputs({"Out"})
    .Attrs({"ratio: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_DropoutWithTrainingSwitch))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_DropoutWithTrainingSwitch))
    .SetKernelFn(PD_KERNEL(Kernel_Function));

PD_BUILD_GRAD_OP(custom_DropoutWithTrainingSwitch)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));

// EmbeddingGather
std::vector<std::vector<int64_t>> InferShape_EmbeddingGather(
    std::vector<int64_t> x_shape, std::vector<int64_t> y_shape) {
    x_shape[0] = y_shape[0];
  return {x_shape};
}
std::vector<paddle::DataType> InferDtype_EmbeddingGather(paddle::DataType x_dtype, paddle::DataType y_dtype) {
  return {x_dtype};
}
// indices is int32/int64
PD_BUILD_OP(custom_EmbeddingGather)
    .Inputs({"X", "indices"})
    .Outputs({"Out"})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_EmbeddingGather))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_EmbeddingGather));

PD_BUILD_GRAD_OP(custom_EmbeddingGather)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));

// Checkpointoutput
std::vector<std::vector<int64_t>> InferShape_Checkpointoutput(
    std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> InferDtype_Checkpointoutput(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(checkpointoutput)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_Checkpointoutput))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_Checkpointoutput));


PD_BUILD_GRAD_OP(checkpointoutput)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));

// nllloss
std::vector<std::vector<int64_t>> InferShape_NllLoss(
    std::vector<int64_t> x_shape, std::vector<int64_t> y_shape, const int& reduction, const std::string& ignoreIndex, const bool& inputIsLogProbability) {
  // 0: sum, 1: mean, 2: none
  if (reduction == 2) {
    return {y_shape};
  } else {
    return {{1}};
  }
}

std::vector<paddle::DataType> InferDtype_NllLoss(paddle::DataType x_dtype, paddle::DataType y_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_nll_loss)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Attrs({"reduction: int", "ignoreIndex: std::string", "inputIsLogProbability: bool"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_NllLoss))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_NllLoss));


PD_BUILD_GRAD_OP(custom_nll_loss)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));

// CastFromFp8
std::vector<std::vector<int64_t>> InferShape_CastFromFp8(
    std::vector<int64_t> x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> InferDtype_CastFromFp8(paddle::DataType x_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(custom_CastFromFp8)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"to: std::string", "nBitMantissa: std::string", "nBitExponent: std::string", "exponentBias: std::string"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_CastFromFp8))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_CastFromFp8));


PD_BUILD_GRAD_OP(custom_CastFromFp8)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));

// CastToFp8
std::vector<std::vector<int64_t>> InferShape_CastToFp8(
    std::vector<int64_t> x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> InferDtype_CastToFp8() {
  return {paddle::DataType::INT8};
}

PD_BUILD_OP(custom_CastToFp8)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"nBitMantissa: std::string", "nBitExponent: std::string", "exponentBias: std::string"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_CastToFp8))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_CastToFp8));


PD_BUILD_GRAD_OP(custom_CastToFp8)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));