// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>

class Add {
public:
    Add(const std::vector<std::size_t> aShape_, const std::vector<std::size_t> bShape_, const std::string& debugStr_);

    poplar::Tensor forward(poplar::Graph &graph,
                           poplar::Tensor a, poplar::Tensor b,
                           poplar::program::Sequence &prog,
                           const std::string &debugStr) const;

    void backward(poplar::Graph &graph,
                  poplar::Tensor dOut,
                  poplar::Tensor &dA, poplar::Tensor &dB,
                  poplar::program::Sequence &prog,
                  const std::string &debugStr) const;

    void forwardInPlace(poplar::Graph &graph,
                        poplar::Tensor aInOut, poplar::Tensor b,
                        poplar::program::Sequence &prog,
                        const std::string &debugStr) const;

private:
    std::vector<std::size_t> aShape;
    std::vector<std::size_t> bShape;
    std::size_t aSqueeze;
    std::size_t bSqueeze;
};