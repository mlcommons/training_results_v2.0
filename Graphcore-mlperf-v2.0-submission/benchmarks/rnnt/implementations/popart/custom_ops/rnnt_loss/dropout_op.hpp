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

//////////////////////////////////////////////////////////////////////////
namespace CustomOperators {
    const popart::OperatorIdentifier CustomDropout = {"com.acme", "CustomDropout", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
    const popart::OperatorIdentifier CustomDropoutGrad = {"com.acme", "CustomDropoutGrad", 1};
} // namespace CustomGradOperators

//////////////////////////////////////////////////////////////////////////
class CustomDropoutOp : public popart::RandomBaseOp {
public:
    CustomDropoutOp(const popart::OperatorIdentifier &_opid,
                    float dropoutRate_,
                    const popart::Op::Settings &settings_,
                    const std::string &debugStr_ = "custom_op/CustomDropoutOp");

    // Op
    void setup() final override;
    std::unique_ptr<popart::Op> clone() const final override;
    std::vector<std::unique_ptr<popart::Op>> getGradOps() final override;
    void appendOutlineAttributes(popart::OpSerialiserBase &) const final override;
    float getSubgraphValue() const final override { return getHighSubgraphValue(); }

    // RandomBaseOp
    popart::InIndex getSeedInIndex() const final override { return SEED_INDEX; }

    enum InputIdxs {
        IN_INDEX,
#if OUTPUT_DROPOUT_MASK
        DUMMY_MASK_INDEX,
#endif
        SEED_INDEX,
    };

    enum OutputIdxs {
        OUT_INDEX,
#if OUTPUT_DROPOUT_MASK
        MASK_INDEX,
#endif
    };

    float getDropoutRate() const { return dropoutRate; }

    const std::string& getDebugStr() const { return debugStr; }

private:
    float dropoutRate;
    std::string debugStr;

public:
    // For debugging purpose only
    const void* const pOp;
};

//////////////////////////////////////////////////////////////////////////
class CustomDropoutGradOp : public popart::RandomBaseOp {
public:
    CustomDropoutGradOp(const CustomDropoutOp &fwdOp_);

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
        D_IN_INDEX,
        SEED_INDEX,
    };

    enum GradOutputIdxs {
        D_OUT_INDEX,
#if OUTPUT_DROPOUT_MASK
        D_DUMMY_MASK_INDEX,
#endif
    };

    float getDropoutRate() const { return dropoutRate; }

    const std::string& getDebugStr() const { return debugStr; }

private:
    float dropoutRate;
    std::string debugStr;

public:
    // For debugging purpose only
    // Save original pointer, so it propagates through all copy constructors
    const void* const pFwdOp;
};

//////////////////////////////////////////////////////////////////////////
class CustomDropoutOpx : public popart::popx::Opx {
public:
    CustomDropoutOpx(popart::Op *op, popart::popx::Devicex *devicex);

    void grow(poplar::program::Sequence &prog) const final override;
};

//////////////////////////////////////////////////////////////////////////
class CustomDropoutGradOpx : public popart::popx::Opx {
public:
    CustomDropoutGradOpx(popart::Op *op, popart::popx::Devicex *devicex);
    
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

}
