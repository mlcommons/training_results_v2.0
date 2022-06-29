// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "dropout.hpp"

#include <popart/error.hpp>
#include <popops/ElementWise.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>

#include "rnnt_utils.hpp"

#include <assert.h>

static LogLevel logLevel = getLogLevel();

namespace pe = popops::expr;

void Dropout::setDropoutRate(float r) {
    assert(r >= 0.0 && r <= 1.0);
    dropoutRate = r;
}

#if OUTPUT_DROPOUT_MASK
std::pair<poplar::Tensor, poplar::Tensor>
#else
poplar::Tensor
#endif
Dropout::forwardOrBackward(poplar::Graph &graph,
                                          poplar::Tensor tIn,
                                          const poplar::Tensor *seed,
                                          poplar::program::Sequence &prog,
                                          const std::string &debugStr) const {
    assert(tIn.shape() == maskTensorInfo.shape);
    if (maskTensorInfo.shape != tIn.shape()) {
        throw popart::error("Dropout::forwardOrBackward: Dropout mask shape is different from the input tensor shape!");
    }

    double dropoutProbability = 1. - static_cast<double>(dropoutRate);
    // When dropout rate is outside of (0,1), an error is thrown in op creation,
    // so we avoid div/0 errors here.
    float scale = 1.f / (1.f - dropoutRate);

    const auto dType = tIn.elementType();

    poplar::Tensor mask = genRandomMask(graph, seed, 0u,
                                        poplar::HALF, maskTensorInfo, dropoutProbability,
                                        prog, debugStr + "/mask");

    // Use the mask to multiply by the input tensor and scale up.
    poplar::Tensor tOut = popops::map(graph,
                                      pe::Mul(pe::Mul(pe::_1, pe::Cast(pe::_2, dType)), pe::Const(scale)),
                                      {tIn, mask},
                                      prog,
                                      debugStr + "/dropout");
#if OUTPUT_DROPOUT_MASK
    return std::pair<poplar::Tensor, poplar::Tensor>(tOut, mask);
#else
    return tOut;
#endif
}

#if OUTPUT_DROPOUT_MASK
    poplar::Tensor
#else
    void 
#endif
Dropout::forwardOrBackwardInPlace(poplar::Graph &graph,
                                  poplar::Tensor inOut,
                                  const poplar::Tensor *seed,
                                  poplar::program::Sequence &prog,
                                  const std::string &debugStr) const {
    if (maskTensorInfo.shape != inOut.shape()) {
        throw popart::error("Dropout::forwardOrBackwardInPlace: Dropout mask shape is different from the input tensor shape!");
    }
    double dropoutProbability = 1. - static_cast<double>(dropoutRate);
    // When dropout rate is outside of (0,1), an error is thrown in op creation,
    // so we avoid div/0 errors here.
    float scale = 1.f / (1.f - dropoutRate);

    const auto dType = inOut.elementType();

    poplar::Tensor mask = genRandomMask(graph, seed, 0u,
                                        poplar::HALF, maskTensorInfo, dropoutProbability,
                                        prog, debugStr + "/mask");

    // Use the mask to multiply by the input tensor and scale up.
    popops::mapInPlace(graph,
                       pe::Mul(pe::Mul(pe::_1, pe::Cast(pe::_2, dType)), pe::Const(scale)),
                       {inOut, mask},
                       prog,
                       debugStr + "/dropoutInPlace");
#if OUTPUT_DROPOUT_MASK
    return mask;
#endif
}

DropoutCache& DropoutCache::instance() {
    static DropoutCache instance;
    return instance;
}

const Dropout& DropoutCache::getDropout(float ratio_, const MaskTensorInfo &maskTensorInfo_) {
    std::string strShape = shapeToStr(maskTensorInfo_.shape);
    auto iter = cacheByShapes.find(strShape);
    if (iter != cacheByShapes.end()) {
        return iter->second;
    }
    if (logLevel >= LogLevel::Trace) {
        printf("DropoutCache::getDropout(): inserted dropout cache for the shape [%s]\n", strShape.c_str());
    }
    auto res = cacheByShapes.insert(std::make_pair(strShape, Dropout(ratio_, maskTensorInfo_)));
    iter = res.first;
    return iter->second;
}

std::string DropoutCache::shapeToStr(const std::vector<std::size_t>& shape) {
    std::string strShape = std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
        strShape.append("x");
        strShape.append(std::to_string(shape[i]));
    }
    return strShape;
}
