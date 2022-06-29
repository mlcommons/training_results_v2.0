

// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "sparse_log_softmax.hpp"

#include "logsoftmax.hpp"
#include <poputil/TileMapping.hpp>

poplar::Tensor SparseLogSoftmax::forward(poplar::Graph &graph,
                                         poplar::Tensor logProbs, poplar::Tensor labels, poplar::Tensor labelLengths,
                                         poplar::program::Sequence &prog,
                                         const std::string &debugStr) const {
    std::size_t alphabet = logProbs.dim(3);
    poputil::mapTensorLinearly(graph, logProbs, 1, alphabet);
    poplar::Tensor compacted = logSoftmaxRnnt(graph, logProbs, labels, labelLengths,
                                    prog, "SparseLogSoftmaxFwd");
    return compacted;
}

void SparseLogSoftmax::backwardInPlace(poplar::Graph &graph,
                                       poplar::Tensor compactedGrads, poplar::Tensor &logProbsInGradsOut, poplar::Tensor labels, poplar::Tensor labelLengths,
                                       poplar::program::Sequence &prog,
                                       const std::string &debugStr) const {
    std::size_t alphabet = logProbsInGradsOut.dim(3);
    poputil::mapTensorLinearly(graph, logProbsInGradsOut, 1, alphabet);
    poputil::mapTensorLinearly(graph, compactedGrads, 1, 2);
    poplar::Tensor gradsOut =
        logSoftmaxRnntGrad(graph, logProbsInGradsOut, compactedGrads, labels,
                           labelLengths, prog, "SparseLogSoftmaxGrad");
    logProbsInGradsOut = gradsOut;
}