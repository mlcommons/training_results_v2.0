// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include <string>
#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>

class SparseLogSoftmax {
public:
    poplar::Tensor forward(poplar::Graph &graph,
                           poplar::Tensor logProbs, poplar::Tensor labels, poplar::Tensor labelLengths,
                           poplar::program::Sequence &prog,
                           const std::string &debugStr) const;

    void backwardInPlace(poplar::Graph &graph,
                         poplar::Tensor compactedGrads, poplar::Tensor &logProbsInGradsOut, poplar::Tensor labels, poplar::Tensor labelLengths,
                         poplar::program::Sequence &prog,
                         const std::string &debugStr) const;
};