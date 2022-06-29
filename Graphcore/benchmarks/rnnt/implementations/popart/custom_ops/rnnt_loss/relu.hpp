// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include <string>
#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>

class Relu {
public:
    void forwardInPlace(poplar::Graph &graph,
                        poplar::Tensor &inOut,
                        poplar::program::Sequence &prog,
                        const std::string &debugStr) const;

    void backwardInPlace(poplar::Graph &graph,
                         poplar::Tensor outAct, poplar::Tensor &dInOut,
                         poplar::program::Sequence &prog,
                         const std::string &debugStr) const;
};