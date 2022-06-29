// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "add.hpp"
#include <popart/logging.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

#include <numeric>

Add::Add(const std::vector<std::size_t> aShape_, const std::vector<std::size_t> bShape_, const std::string& debugStr)
    : aShape(aShape_)
    , bShape(bShape_)
    , aSqueeze(0)
    , bSqueeze(0)
{
    if (popart::logging::shouldLog(popart::logging::Module::none, popart::logging::Level::Trace)) {
        std::string shapeAFmt;
        for (size_t i = 0; i < aShape_.size(); ++i) {
            shapeAFmt += std::to_string(aShape_[i]);
            if (i < aShape_.size() - 1) {
                shapeAFmt += " x ";
            }
        }
        std::string shapeBFmt;
        for (size_t i = 0; i < bShape_.size(); ++i) {
            shapeBFmt += std::to_string(bShape_[i]);
            if (i < bShape_.size() - 1) {
                shapeBFmt += " x ";
            }
        }
        popart::logging::trace("{}/Add(): a shape = [{}], b shape = [{}]", debugStr, shapeAFmt, shapeBFmt);
    }

    while (aShape.size() < bShape.size()) {
        aShape.insert(aShape.begin(), 1u);
        ++aSqueeze;
    }
    while (aShape.size() > bShape.size()) {
        bShape.insert(bShape.begin(), 1u);
        ++bSqueeze;
    }
    for (size_t i = 0; i < aShape.size(); ++i) {
        assert(aShape[i] == 1 || bShape[i] == 1 || aShape[i] == bShape[i]);
    }
}

static void expandToRank(poplar::Tensor &t, std::size_t rank) {
    assert(t.rank() <= rank);
    if (t.rank() == rank) {
        return;
    }

    std::vector<std::size_t> dimsExpand(rank - t.rank(), 0);
    t = t.expand(dimsExpand);
    assert(t.rank() == rank);
}

poplar::Tensor Add::forward(poplar::Graph &graph,
                            poplar::Tensor a, poplar::Tensor b,
                            poplar::program::Sequence &prog,
                            const std::string &debugStr) const {
    expandToRank(a, aShape.size());
    expandToRank(b, bShape.size());
    assert(a.shape() == aShape);
    assert(b.shape() == bShape);

    return popops::add(graph, a, b, prog, debugStr);
}

void Add::forwardInPlace(poplar::Graph &graph,
                         poplar::Tensor aInOut, poplar::Tensor b,
                         poplar::program::Sequence &prog,
                         const std::string &debugStr) const {
    assert(aInOut.rank() == aShape.size());
    expandToRank(b, bShape.size());
    for (size_t i = 0; i < aShape.size(); ++i) {
        assert(bShape[i] == 1 || aShape[i] == bShape[i]);
    }

    popops::addInPlace(graph, aInOut, b, prog, debugStr);
}

void Add::backward(poplar::Graph &graph,
                   poplar::Tensor dOut,
                   poplar::Tensor &dA, poplar::Tensor &dB,
                   poplar::program::Sequence &prog,
                   const std::string &debugStr) const {
    std::vector<std::size_t> dimsReduceA;
    std::vector<std::size_t> dimsReduceB;
    std::vector<std::size_t> dimsExpandA;
    std::vector<std::size_t> dimsExpandB;
    const auto &dOutShape = dOut.shape();

    for (size_t i = 0, nReducedA = 0, nReducedB = 0; i < aShape.size(); ++i) {
        if (aShape[i] == 1 && bShape[i] > 1) {
            assert(dOutShape[i] == bShape[i]);
            dimsReduceA.push_back(i);
            dimsExpandA.push_back(nReducedA);
            ++nReducedB;
        } else if (aShape[i] > 1 && bShape[i] == 1) {
            assert(dOutShape[i] == aShape[i]);
            dimsReduceB.push_back(i);
            dimsExpandB.push_back(nReducedB);
            ++nReducedA;
        } else {
            assert(dOutShape[i] == aShape[i]);
            ++nReducedA;
            ++nReducedB;
        }
    }
    std::vector<std::size_t> dimsSqueezeA(aSqueeze, 0);
    std::iota(dimsSqueezeA.begin(), dimsSqueezeA.end(), 0);
    std::vector<std::size_t> dimsSqueezeB(bSqueeze, 0);
    std::iota(dimsSqueezeB.begin(), dimsSqueezeB.end(), 0);

    if (!dimsReduceA.empty()) {
        dA = popops::reduce(graph, dOut, dimsReduceA, popops::Operation::ADD, prog, debugStr);
        dA = dA.expand(dimsExpandA);
    } else {
        dA = dOut;
    }
    if (!dimsReduceB.empty()) {
        dB = popops::reduce(graph, dOut, dimsReduceB, popops::Operation::ADD, prog, debugStr);
        dB = dB.expand(dimsExpandB);
    } else {
        dB = dOut;
    }
    assert(dA.shape() == aShape);
    assert(dB.shape() == bShape);

    if (!dimsSqueezeA.empty()) {
        dA = dA.squeeze(dimsSqueezeA);
    }
    if (!dimsSqueezeB.empty()) {
        dB = dB.squeeze(dimsSqueezeB);
    }

    if (popart::logging::shouldLog(popart::logging::Module::none, popart::logging::Level::Trace)) {
        std::string shapeAFmt = std::to_string(dA.dim(0));
        for (size_t i = 1; i < dA.rank(); ++i) {
            shapeAFmt += " x " + std::to_string(dA.dim(i));
        }
        std::string shapeBFmt = std::to_string(dB.dim(0));
        for (size_t i = 1; i < dB.rank(); ++i) {
            shapeBFmt += " x " + std::to_string(dB.dim(i));
        }
        popart::logging::trace("{}/Add::backward: dA.shape = [{}], dB.shape = [{}]", debugStr, shapeAFmt, shapeBFmt);
    }
}