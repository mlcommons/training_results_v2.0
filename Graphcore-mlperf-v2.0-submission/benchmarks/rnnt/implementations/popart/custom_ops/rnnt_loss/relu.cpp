// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "relu.hpp"
#include <popnn/NonLinearity.hpp>

void Relu::forwardInPlace(poplar::Graph &graph,
                          poplar::Tensor &inOut,
                          poplar::program::Sequence &prog,
                          const std::string &debugStr) const {
        popnn::reluInPlace(graph, inOut, prog, debugStr);
}

void Relu::backwardInPlace(poplar::Graph &graph,
                           poplar::Tensor outAct, poplar::Tensor &dInOut,
                           poplar::program::Sequence &prog,
                           const std::string &debugStr) const {
        poplar::Tensor dOut = popnn::nonLinearityInputGradient(graph, popnn::NonLinearityType::RELU, outAct, dInOut, prog, debugStr);
        dInOut = dOut;
}