// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include <string>
#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>

class MatMul {
public:
    MatMul(const std::vector<std::size_t> lhsShape_, const std::vector<std::size_t> rhsbShape_);

    poplar::Tensor forward(poplar::Graph &graph,
                           poplar::Tensor lhsTensor, poplar::Tensor rhsTensor,
                           poplin::matmul::PlanningCache *matmulCache,
                           poplar::program::Sequence &prog,
                           const std::string &debugStr) const;

    void backward(poplar::Graph &graph,
                  poplar::Tensor dOut, poplar::Tensor fwdLhs, poplar::Tensor fwdRhs,
                  poplar::Tensor &dFwdLhs, poplar::Tensor &dFwdRhs,
                  poplin::matmul::PlanningCache *matmulCache,
                  poplar::program::Sequence &prog,
                  const std::string &debugStr) const;

    poplar::Tensor createInput(poplar::Graph &graph, const poplar::Type &rhsType,
                               poplin::matmul::PlanningCache *matmulCache, const std::string &name) const;

    unsigned getSerializationFactor() const {
        return serializationFactor;
    }
    
    void setSerializationFactor(unsigned serializationFactor_) {
        serializationFactor = serializationFactor_;
    }

private:
    std::vector<std::size_t> lhsShape;
    std::vector<std::size_t> rhsShape;

    bool transposeRhs = false;
    unsigned serializationFactor = 0;
};