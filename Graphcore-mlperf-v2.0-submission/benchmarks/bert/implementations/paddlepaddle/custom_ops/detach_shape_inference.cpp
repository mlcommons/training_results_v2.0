// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/shapeinference.hpp>
#include <popart/logging.hpp>

auto detachShapeInferenceFun = [](popart::ShapeInferenceContext &ctx) {
  popart::logging::info("enter detachShapeInferenceFun");
  ctx.outInfo(0) = ctx.inInfo(0);
  popart::logging::info("leave detachShapeInferenceFun");
};

static popart::RegisterShapeInferenceFunction
    detachShapeInferenceFunction({"custom.ops", "Detach", 1},
                       detachShapeInferenceFun);
