// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include "poplar/Tensor.hpp"

#include "poplar/Tensor.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"

#include <string>

struct MaskTensorInfo {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<std::vector<poplar::Interval>> tileMapping;
};

poplar::Tensor genRandomMask(poplar::Graph &graph, const poplar::Tensor *masterSeed, uint32_t seedModifier,
                             poplar::Type type, const MaskTensorInfo &info, double prob,
                             poplar::program::Sequence &prog, const poplar::DebugContext &debugContext);