// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once
#include <popart/opidentifier.hpp>
#include <popart/op.hpp>
#include <popart/op/randombase.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>

#include <memory>
#include <vector>
#include <map>

#include "add.hpp"
#include "relu.hpp"
#include "dropout.hpp"
#include "matmul.hpp"
#include "sparse_log_softmax.hpp"

namespace CustomOperators {
    const popart::OperatorIdentifier GetCompactLogProbs = {"com.acme", "GetCompactLogProbs", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
    const popart::OperatorIdentifier GetCompactLogProbsGrad = {"com.acme", "GetCompactLogProbsGrad", 1};
} // namespace CustomGradOperators

//////////////////////////////////////////////////////////////////////////
class GetCompactLogProbsOp : public popart::RandomBaseOp {
public:
    GetCompactLogProbsOp(const popart::OperatorIdentifier &_opid,
               const popart::Op::Settings &settings_,
               const std::string &debugStr_ = "custom_op/GetCompactLogProbsOp");

    // Op
    void setup() final override;
    std::unique_ptr<popart::Op> clone() const final override;
    std::vector<std::unique_ptr<popart::Op>> getGradOps() final override;
    void appendOutlineAttributes(popart::OpSerialiserBase &) const final override;
    float getSubgraphValue() const final override { return getHighSubgraphValue(); }

    // RandomBaseOp
    popart::InIndex getSeedInIndex() const final override { return SEED_INDEX; }

    enum InputIdxs {
        TRANS_IN_INDEX,
        PRED_IN_INDEX,
        FCJOIN_W_IN_INDEX,
        FCJOIN_B_IN_INDEX,
        LABELS_IN_INDEX,
        LABELS_LEN_IN_INDEX,
        SEED_INDEX,
    };

    enum OutputIdxs {
        PROBS_OUT_INDEX,
#if NO_RECOMPUTE_FC_IN
        FC_INPUT_OUT_INDEX,
#endif
#if NO_RECOMPUTE_FC_OUT
        LOGITS_OUT_INDEX,
#endif
#if OUTPUT_DROPOUT_MASK
        MASK_INDEX,
#endif
    };

    const std::string& getDebugStr() const { return debugStr; }
    float getDropoutRate() const { return dropoutRate; }
    unsigned getShiftLabelsBy1() const { return shiftLabelsBy1; }

    std::shared_ptr<Add> getAdd() const { return add; }
    std::shared_ptr<Relu> getRelu() const { return relu; }
    std::shared_ptr<MatMul> getFcMatMul() const { return fcMatmul; }
    std::shared_ptr<Add> getFcAddBias() const { return fcAddBias; }
    std::shared_ptr<SparseLogSoftmax> getSparseLogSoftmax() const { return sparseLogSoftmax; }

    void setSerializationFactor(unsigned serializationFactor_) {
        serializationFactor = serializationFactor_;
    }

    void setSplitFactor(unsigned splitFactor_) {
        splitFactor = splitFactor_;
    }

    void setDropoutRate(float rate_) {
        dropoutRate = rate_;
    }

    void setShiftLabelsBy1(unsigned shiftLabelsBy1_) {
        shiftLabelsBy1 = shiftLabelsBy1_;
    }

private:
    std::shared_ptr<Add> add;
    std::shared_ptr<Relu> relu;
    std::shared_ptr<MatMul> fcMatmul;
    std::shared_ptr<Add> fcAddBias;
    std::shared_ptr<SparseLogSoftmax> sparseLogSoftmax;
    unsigned serializationFactor = 0;
    unsigned splitFactor = 0;
    float dropoutRate = 0.0f;
    unsigned shiftLabelsBy1 = 1u;
    std::string debugStr;
};

//////////////////////////////////////////////////////////////////////////
class GetCompactLogProbsGradOp : public popart::RandomBaseOp {
public:
    GetCompactLogProbsGradOp(const GetCompactLogProbsOp &fwdOp_);

    // Op
    std::unique_ptr<popart::Op> clone() const final override;
    void setup() final override;
    const std::vector<popart::GradInOutMapper> &gradInputInfo() const final override;
    const std::map<int, int> &gradOutToNonGradIn() const final override;
    void appendOutlineAttributes(popart::OpSerialiserBase &) const final override;
    float getSubgraphValue() const final override { return getHighSubgraphValue(); }

    // RandomBaseOp
    popart::InIndex getSeedInIndex() const final override { return SEED_INDEX; }

    enum GradInputIdxs {
        D_OUTPUT_IN_INDEX,
        FCJOIN_W_IN_INDEX,
#if !NO_RECOMPUTE_FC_OUT
        FCJOIN_B_IN_INDEX,
#endif
        LABELS_IN_INDEX,
        LABELS_LEN_IN_INDEX,
#if NO_RECOMPUTE_FC_IN
        FC_INPUT_OUTPUT_IN_INDEX,
#else
        TRANS_IN_INDEX,
        PRED_IN_INDEX,
#endif
#if NO_RECOMPUTE_FC_OUT
        LOGITS_OUTPUT_IN_INDEX,
#endif
        SEED_INDEX
    };

    enum GradOutputIdxs {
        D_TRANS_OUT_INDEX,
        D_PRED_OUT_INDEX,
        D_FCJOIN_W_OUT_INDEX,
        D_FCJOIN_B_OUT_INDEX,
    };

    const std::string& getDebugStr() const { return debugStr; }
    float getDropoutRate() const { return dropoutRate; }
    unsigned getShiftLabelsBy1() const { return shiftLabelsBy1; }

    std::shared_ptr<Add> getAdd() const { return add; }
    std::shared_ptr<Relu> getRelu() const { return relu; }
    std::shared_ptr<MatMul> getFcMatMul() const { return fcMatmul; }
    std::shared_ptr<Add> getFcAddBias() const { return fcAddBias; }
    std::shared_ptr<SparseLogSoftmax> getSparseLogSoftmax() const { return sparseLogSoftmax; }

public:
    // For debugging purpose only
    const void* const pOp;

private:
    popart::TensorInfo fwdTransInfo;
    popart::TensorInfo fwdPredInfo;
    popart::TensorInfo fwdJoinFcWInfo;
    popart::TensorInfo fwdJoinFcBInfo;
    std::string debugStr;
    std::shared_ptr<Add> add;
    std::shared_ptr<Relu> relu;
    std::shared_ptr<MatMul> fcMatmul;
    std::shared_ptr<Add> fcAddBias;
    std::shared_ptr<SparseLogSoftmax> sparseLogSoftmax;
    float dropoutRate;
    unsigned shiftLabelsBy1 = 1u;
};

//////////////////////////////////////////////////////////////////////////
class GetCompactLogProbsOpx : public popart::popx::Opx {
public:
    GetCompactLogProbsOpx(popart::Op *op, popart::popx::Devicex *devicex);

    void grow(poplar::program::Sequence &prog) const final override;
    poplar::Tensor createInput(popart::InIndex index, const poplar::DebugNameAndId &dnai) const final override;
    popart::popx::InputCreatorType getInputCreatorType(popart::InIndex index) const final override;
    bool createsEquiv(int, const popart::popx::Opx *, int) const final override;
    std::set<popart::TensorId> mustExistBeforeCreate(popart::InIndex index0) const final override;
};

//////////////////////////////////////////////////////////////////////////
class GetCompactLogProbsGradOpx : public popart::popx::Opx {
public:
    GetCompactLogProbsGradOpx(popart::Op *op, popart::popx::Devicex *devicex);
    void grow(poplar::program::Sequence &prog) const final override;
};

//////////////////////////////////////////////////////////////////////////
extern "C" {

int outputDropoutMask() {
#if OUTPUT_DROPOUT_MASK
    return 1;
#else
    return 0;
#endif
}

int noRecomputeFcIn() {
#if NO_RECOMPUTE_FC_IN
    return 1;
#else
    return 0;
#endif
}

int noRecomputeFcOut() {
#if NO_RECOMPUTE_FC_OUT
    return 1;
#else
    return 0;
#endif
}

}