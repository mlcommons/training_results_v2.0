// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "matmul.hpp"

#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/popx/poplaroptionsx.hpp>

#include <vector>

// inChanSplit.serial will correspond to serialising over K
// outChanSplit.serial will correspond to serialising over N
// There is no serialisation over M yet

static std::string BuildPlanConstraints(unsigned serializationFactor, bool inPutSer) {
    std::string planConstraints = "";
    if (serializationFactor > 0) {
        planConstraints =
            "{"
            "   \"0\": {"
            "        \"partition\": {";
        if (inPutSer > 0) {
            planConstraints +=
            "           \"inChanSplit\": {";
        } else {
            planConstraints +=
            "           \"outChanSplit\": {";
        }
        planConstraints +=
            "               \"serial\": " + std::to_string(serializationFactor) +
            "           }"
            "       }"
            "   }"
            "}";
    }
    return planConstraints;
}

MatMul::MatMul(const std::vector<std::size_t> lhsShape_, const std::vector<std::size_t> rhsShape_) {
    lhsShape = lhsShape_;
    assert(rhsShape_.size() == 2);
    if (!transposeRhs) {       
        rhsShape = rhsShape_;
    } else {
        rhsShape = std::vector<std::size_t>(rhsShape_[1], rhsShape_[0]);
    }
    if (lhsShape[lhsShape.size() - 1] != rhsShape[0]) {
        throw popart::error("MatMul: common dimensions of matmul lhs and rhs tensors must match !");
    }
}

poplar::Tensor MatMul::forward(poplar::Graph &graph,
                               poplar::Tensor lhs, poplar::Tensor rhs,
                               poplin::matmul::PlanningCache *matmulCache,
                               poplar::program::Sequence &prog,
                               const std::string &debugStr0) const {
    assert(lhs.rank() >= 2);
    assert(rhs.rank() == 2);
    unsigned lhsRank = lhs.rank();
    const auto& lhsShape = lhs.shape();
    unsigned lhsRows = lhsShape[lhsRank - 2];
    unsigned lhsCols = lhsShape[lhsRank - 1];

    if (transposeRhs) {
        rhs = rhs.transpose();
    }
    unsigned rhsCols = rhs.shape()[1];

    std::vector<size_t> inputGroupDims;
    unsigned numGroupElems = 1;
    for (size_t i = 0; i + 2 < lhsRank; i++) {
        inputGroupDims.push_back(lhsShape[i]);
        numGroupElems *= lhsShape[i];
    }
    lhs = lhs.reshape({numGroupElems * lhsRows, lhsCols});

    unsigned mDim = lhs.shape()[0];
    unsigned kDim = lhs.shape()[1];
    unsigned nDim = rhs.shape()[1];
    popart::logging::trace("MatMul::forward: m = {}, k = {}, n = {}", mDim, kDim, nDim);

    unsigned serFactor = serializationFactor;
    bool inPutSer = true;
    unsigned serDim = kDim;
    if (kDim < nDim) {
        inPutSer = false;
        serDim = nDim;
    }
    
    popart::popx::PoplarOptions opts;
    opts.options["fullyConnectedPass"] = "TRAINING_FWD";    // Hard-coded for training now
    opts.options["partialsType"] = "half";

    if (serFactor > 1) {
        if (serDim % serFactor > 0) {
            popart::logging::warn("MatMul::forward: this = {}, fw: Cannot serialize dimension of size {} to a factor of {}", this, serDim, serFactor);
            serFactor = 0;
        } else {
            popart::logging::trace("MatMul::forward: this = {}, fw: Serializing {} dimension of size {} to a factor of {}", this, inPutSer ? "K" : "N", serDim, serFactor);
        }
    }
    if (serFactor > 1) {
        std::string planConstraints = BuildPlanConstraints(serFactor, inPutSer);
        opts.options["planConstraints"] = planConstraints;
        popart::logging::trace("MatMul::forward: this = {}, fw: planConstraints = {}", this, planConstraints.c_str());
    }
    poplar::OptionFlags mmOpt = opts.toOptionFlags();

    std::string debugStr = debugStr0
         + "_" + std::to_string(lhs.shape()[0])
         + "x" + std::to_string(lhs.shape()[1])
         + "x" + std::to_string(rhs.shape()[1]);

    poplar::Tensor out = poplin::matMul(graph, lhs, rhs, prog,
                         debugStr, mmOpt, matmulCache);

    std::vector<size_t> outShape = inputGroupDims;
    outShape.push_back(lhsRows);
    outShape.push_back(rhsCols);
    out = out.reshape(outShape);
    return out;
}

void MatMul::backward(poplar::Graph &graph,
                      poplar::Tensor dOut, poplar::Tensor fwdLhs, poplar::Tensor fwdRhs,
                      poplar::Tensor &dFwdLhs, poplar::Tensor &dFwdRhs,
                      poplin::matmul::PlanningCache *matmulCache,
                      poplar::program::Sequence &prog,
                      const std::string &debugStr0) const {
    assert(dOut.rank() == fwdLhs.rank());
    assert(fwdLhs.rank() >= 2);
    assert(fwdRhs.rank() == 2);
    unsigned fwdLhsRank = fwdLhs.rank();
    const auto& fwdLhsShape = fwdLhs.shape();
    const auto& dOutShape = dOut.shape();
    const auto& fwdRhsShape = fwdRhs.shape();

    std::vector<size_t> inputGroupDims;
    unsigned numGroupElems = 1;
    for (size_t i = 0; i + 2 < fwdLhsRank; i++) {
        inputGroupDims.push_back(fwdLhsShape[i]);
        numGroupElems *= fwdLhsShape[i];
        assert(dOutShape[i] == fwdLhsShape[i]);
    }
    unsigned fwdLhsRows = fwdLhsShape[fwdLhsRank - 2];
    unsigned fwdLhsCols = fwdLhsShape[fwdLhsRank - 1];   
    unsigned dOutRows = dOutShape[fwdLhsRank - 2];
    unsigned dOutCols = dOutShape[fwdLhsRank - 1];
    assert(fwdLhsRows == dOutRows);
    unsigned dFwdLhsRows = fwdLhsRows;
    unsigned dFwdLhsCols = !transposeRhs ? fwdRhsShape[0] : fwdRhsShape[1];

    fwdLhs = fwdLhs.reshape({numGroupElems * fwdLhsRows, fwdLhsCols});
    dOut = dOut.reshape({numGroupElems * dOutRows, dOutCols});

    poplar::Tensor fwdRhsT = transposeRhs ? fwdRhs : fwdRhs.transpose();

    unsigned batchDim = dOut.shape()[0];
    unsigned outDim = dOut.shape()[1];
    unsigned inDim = fwdRhsT.shape()[1];

    unsigned mDim, kDim, nDim;
    unsigned serFactor;
    bool inPutSer;
    unsigned serDim;
    
    // bw
    popart::popx::PoplarOptions opts;
    opts.options["fullyConnectedPass"] = "TRAINING_BWD";
    opts.options["partialsType"] = "half";

    mDim = batchDim;
    kDim = outDim;
    nDim = inDim;

    popart::logging::trace("MatMul::backward: m = {}, k = {}, n = {}", mDim, kDim, nDim);

    serFactor = serializationFactor;
    inPutSer = true;
    serDim = kDim;
    if (kDim < nDim) {
        inPutSer = false;
        serDim = nDim;
    }
    if (serFactor > 1) {
        if (serDim % serFactor > 0) {
            popart::logging::warn("MatMul::backward: this = {}, bw: Cannot serialize dimension of size {} to a factor of {}", this, serDim, serFactor);
            serFactor = 0;
        } else {
            popart::logging::trace("MatMul::backward: this = {}, bw: Serializing {} dimension of size {} to a factor of {}", this, inPutSer ? "K" : "N", serDim, serFactor);
        }
    }
    if (serFactor > 1) {
        std::string planConstraints = BuildPlanConstraints(serFactor, inPutSer);
        opts.options["planConstraints"] = planConstraints;
        popart::logging::trace("MatMul::backward: this = {}, bw: planConstraints = {}", this, planConstraints.c_str());
    }
    poplar::OptionFlags mmOpt = opts.toOptionFlags();

    std::string debugStr = debugStr0 + "_bw" +
         + "_" + std::to_string(dOut.shape()[0])
         + "x" + std::to_string(dOut.shape()[1])
         + "x" + std::to_string(fwdRhsT.shape()[1]);
    popart::logging::debug("{}", debugStr.c_str());

    
    dFwdLhs = poplin::matMul(graph, dOut, fwdRhsT, prog,
                                   debugStr, mmOpt, matmulCache);

    std::vector<size_t> dFwdLhsShape = inputGroupDims;
    dFwdLhsShape.push_back(dFwdLhsRows);
    dFwdLhsShape.push_back(dFwdLhsCols);
    dFwdLhs = dFwdLhs.reshape(dFwdLhsShape);

    // wu
    opts.options["fullyConnectedPass"] = "TRAINING_WU";
    opts.options["partialsType"] = "half";

    enum DimIdx {BatchDimIdx, InDimIdx, OutDimIdx};

    // in batch, in, out order
    DimIdx maxDimIdx = BatchDimIdx;
    unsigned maxDim = batchDim;
    if (inDim > maxDim) {
        maxDimIdx = InDimIdx;
        maxDim = inDim;
    }
    if (outDim > maxDim) {
        maxDimIdx = OutDimIdx;
        maxDim = outDim;
    }

    serFactor = serializationFactor;
    // Normal wu pass
    // Xt[I,B] x dY[B,O] = dW[I,O]
    // I x B x O
    //
    // Transposed wu pass
    // dYt[O,B] x X[B,I] = dWt[O,I]
    // O x B x I 
    //
    bool transposeMatMul = false;
    bool transposeResult = false;
    if (!transposeRhs) {
        if (serFactor > 0 && maxDimIdx == InDimIdx) {
            // Cannot serialize across M currently
            transposeMatMul = true;
            transposeResult = true;
        }
    } else {
        transposeMatMul = true;
        if (serFactor > 0 && maxDimIdx == OutDimIdx) {
            // Cannot serialize across M currently
            transposeMatMul = false;
            transposeResult = true;
        }
    }
    popart::logging::trace("GetCompactLogProbsGradOpx: this = {}, wu: transposed rhs: {}, transpose matmul: {}, transpose result: {}",
        this, transposeRhs, transposeMatMul, transposeResult);

    if (!transposeMatMul) {
        mDim = inDim;
        kDim = batchDim;
        nDim = outDim;
    } else {
        mDim = outDim;
        kDim = batchDim;
        nDim = inDim;
    }
    inPutSer = true;
    serDim = kDim;
    if (kDim < nDim) {
        inPutSer = false;
        serDim = nDim;
    }
    if (serFactor > 1) {
        if (serDim % serFactor > 0) {
            popart::logging::warn("MatMul::backward: this = {}, wu: Cannot serialize dimension of size {} to a factor of {}", this, serDim, serFactor);
            serFactor = 0;
        } else {
            popart::logging::trace("MatMul::backward: this = {}, wu: Serializing {} dimension of size {} to a factor of {}", this, inPutSer ? "K" : "N", serDim, serFactor);
        }
    }
    if (serFactor > 1) {
        std::string planConstraints = BuildPlanConstraints(serFactor, inPutSer);
        opts.options["planConstraints"] = planConstraints;
        popart::logging::trace("MatMul::backward: this = {}, wu: planConstraints = {}", this, planConstraints.c_str());
    }
    mmOpt = opts.toOptionFlags();
    if (!transposeMatMul) {
        poplar::Tensor fwdLhsT = fwdLhs.transpose();

        debugStr = debugStr0 + "_wu" +
             + "_" + std::to_string(fwdLhsT.shape()[0])
             + "x" + std::to_string(fwdLhsT.shape()[1])
             + "x" + std::to_string(dOut.shape()[1]);
        popart::logging::debug("{}", debugStr.c_str());

        dFwdRhs = poplin::matMul(graph, fwdLhsT, dOut, prog,
                     debugStr, mmOpt, matmulCache);
    } else {
        poplar::Tensor dOutT = dOut.transpose();

        debugStr = debugStr0 + "_wu(t)" +
             + "_" + std::to_string(dOutT.shape()[0])
             + "x" + std::to_string(dOutT.shape()[1])
             + "x" + std::to_string(fwdLhs.shape()[1]);
        popart::logging::debug("{}", debugStr.c_str());

        dFwdRhs = poplin::matMul(graph, dOutT, fwdLhs, prog,
                     debugStr, mmOpt, matmulCache);
    }
    if (transposeResult) {
        dFwdRhs = dFwdRhs.transpose();
    }
    assert(dFwdRhs.shape() == fwdRhsShape);
}

poplar::Tensor MatMul::createInput(poplar::Graph &graph, const poplar::Type &rhsType,
                                   poplin::matmul::PlanningCache *matmulCache, const std::string &name) const {
    unsigned lhsRank = lhsShape.size();
    unsigned lhsRows = lhsShape[lhsRank - 2];
    unsigned lhsCols = lhsShape[lhsRank - 1];

    assert(rhsShape.size() == 2);
    assert(lhsCols == rhsShape[0]);

    unsigned numGroupElems = 1;
    for (size_t i = 0; i + 2 < lhsRank; i++) {
        numGroupElems *= lhsShape[i];
    }
    std::vector<std::size_t> lhsShape1 = {numGroupElems * lhsRows, lhsCols};

    popart::popx::PoplarOptions opts;
    opts.options["fullyConnectedPass"] = "TRAINING_FWD";    // Hard-coded for training now
    poplar::OptionFlags mmOpt = opts.toOptionFlags();
    poplar::Tensor rhs = poplin::createMatMulInputRHS(graph,
                                        rhsType,
                                        lhsShape1,
                                        rhsShape,
                                        name + "/rhs",
                                        mmOpt,
                                        matmulCache);
    if (transposeRhs) {
        rhs = rhs.transpose();
    }
    return rhs;
}
