// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include <string>
#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>

#include "random_mask.hpp"

#include <unordered_map>
#include <string>
#include <vector>

class Dropout {
public:
    Dropout() {}
    Dropout(float dropoutRate_, const MaskTensorInfo &maskTensorInfo_)
        : dropoutRate(dropoutRate_)
        , maskTensorInfo(maskTensorInfo_)
    {}

#if OUTPUT_DROPOUT_MASK
    std::pair<poplar::Tensor, poplar::Tensor>
#else
    poplar::Tensor
#endif
    forwardOrBackward(poplar::Graph &graph,
                                     poplar::Tensor tIn,
                                     const poplar::Tensor *seed,
                                     poplar::program::Sequence &prog,
                                     const std::string &debugStr) const;
#if OUTPUT_DROPOUT_MASK
    poplar::Tensor
#else
    void 
#endif
    forwardOrBackwardInPlace(poplar::Graph &graph,
                             poplar::Tensor t,
                             const poplar::Tensor *seed,
                             poplar::program::Sequence &prog,
                             const std::string &debugStr) const;

    float getDropoutRate() const { return dropoutRate; }
    void setDropoutRate(float r);

    MaskTensorInfo& getMaskTensorInfo() { return maskTensorInfo; }
    const MaskTensorInfo& getMaskTensorInfo() const { return maskTensorInfo; }

    void setMaskTensorInfo(const MaskTensorInfo& ti) { maskTensorInfo = ti; }

private:
    float dropoutRate;
    MaskTensorInfo maskTensorInfo;
};

class DropoutCache {
private:
    DropoutCache() = default;

public:
    static DropoutCache& instance();

    const Dropout& getDropout(float ratio_, const MaskTensorInfo &maskTensorInfo_);

    static std::string shapeToStr(const std::vector<std::size_t>& shape);

private:
    std::unordered_map<std::string, Dropout> cacheByShapes;
};