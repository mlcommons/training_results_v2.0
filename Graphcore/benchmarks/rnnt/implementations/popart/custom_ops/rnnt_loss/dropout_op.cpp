// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "dropout_op.hpp"
#include "dropout.hpp"
#include "rnnt_utils.hpp"

#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/popx/poplaroptionsx.hpp>

using namespace popart;

static LogLevel logLevel = getLogLevel();

static const char* const DEBUG_STR = "debug_str";
static const char* const DROPOUT_RATE = "dropout_rate";

//////////////////////////////////////////////////////////////////////////
CustomDropoutOp::CustomDropoutOp(const OperatorIdentifier &opid_,
                                 float dropoutRate_,
                                 const Op::Settings &settings_,
                                 const std::string &debugStr_)
    : RandomBaseOp(opid_, OptionalDataType(), settings_)
    , dropoutRate(dropoutRate_)
    , debugStr(debugStr_)
    , pOp(this) {
    if (logLevel >= LogLevel::Trace)
        printf("CustomDropoutOp(): this = %p\n", this);
}

std::unique_ptr<Op> CustomDropoutOp::clone() const {
    return std::make_unique<CustomDropoutOp>(*this);
}

void CustomDropoutOp::setup() {
    outInfo(OUT_INDEX) = inInfo(IN_INDEX);
#if OUTPUT_DROPOUT_MASK
    outInfo(MASK_INDEX).set(DataType::FLOAT16, inInfo(IN_INDEX).shape());
#endif
}

std::vector<std::unique_ptr<Op>> CustomDropoutOp::getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<CustomDropoutGradOp>(*this));
    return upops;
}

void CustomDropoutOp::appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute(DROPOUT_RATE, dropoutRate);
    os.appendAttribute(DEBUG_STR, debugStr);
}

//////////////////////////////////////////////////////////////////////////
CustomDropoutGradOp::CustomDropoutGradOp(const CustomDropoutOp &fwdOp)
    : RandomBaseOp(CustomGradOperators::CustomDropoutGrad, OptionalDataType(), fwdOp.getSettings())
    , dropoutRate(fwdOp.getDropoutRate())
    , debugStr(fwdOp.getDebugStr())
    , pFwdOp(fwdOp.pOp) {
    if (logLevel >= LogLevel::Trace)
        printf("CustomDropoutGradOp(): this = %p, fwdOp = %p\n", this, fwdOp.pOp);
}

std::unique_ptr<Op> CustomDropoutGradOp::clone() const {
    return std::make_unique<CustomDropoutGradOp>(*this);
}

void CustomDropoutGradOp::setup() {
    outInfo(D_OUT_INDEX) = inInfo(D_IN_INDEX);
#if OUTPUT_DROPOUT_MASK
    outInfo(D_DUMMY_MASK_INDEX).set(DataType::FLOAT16, inInfo(D_IN_INDEX).shape());
#endif
}

const std::vector<GradInOutMapper> &CustomDropoutGradOp::gradInputInfo() const {
    static const std::vector<GradInOutMapper> inInfo = {
      {D_IN_INDEX, CustomDropoutOp::OUT_INDEX, GradOpInType::GradOut},
      {SEED_INDEX, CustomDropoutOp::SEED_INDEX, GradOpInType::In},
    };
    return inInfo;
}

const std::map<int, int> &CustomDropoutGradOp::gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {
      {D_OUT_INDEX, CustomDropoutOp::IN_INDEX},
#if OUTPUT_DROPOUT_MASK
      {D_DUMMY_MASK_INDEX, CustomDropoutOp::DUMMY_MASK_INDEX},
#endif
    };
    return outInfo;
}

void CustomDropoutGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute(DROPOUT_RATE, dropoutRate);
    os.appendAttribute(DEBUG_STR, debugStr);
}

//////////////////////////////////////////////////////////////////////////
CustomDropoutOpx::CustomDropoutOpx(Op *op, popx::Devicex *devicex)
    : popx::Opx(op, devicex) {
    verifyOp<CustomDropoutOp>(op, CustomOperators::CustomDropout);
    if (logLevel >= LogLevel::Trace) {
        CustomDropoutOp& dropoutOp = getOp<CustomDropoutOp>();
        printf("CustomDropoutOpx(): this = %p, op = %p, origin = %p\n", this, op, dropoutOp.pOp);
    }
}

void CustomDropoutOpx::grow(poplar::program::Sequence &prog) const {
    if (logLevel >= LogLevel::Debug)
        printf("CustomDropoutOpx: grow() this = %p, op = %p", this, op_p);
    CustomDropoutOp& op = getOp<CustomDropoutOp>();
    if (logLevel >= LogLevel::Debug)
        printf(", origin = %p\n", op.pOp);
    const poplar::Tensor& inOut = getInTensor(CustomDropoutOp::IN_INDEX);
    const poplar::Tensor& seed = getInTensor(CustomDropoutOp::SEED_INDEX);

    const Dropout& dropout = DropoutCache::instance().getDropout(op.getDropoutRate(), MaskTensorInfo{"DropoutMask", inOut.shape(), graph().getTileMapping(inOut)});
    if (dropout.getMaskTensorInfo().shape != inOut.shape()) {
        throw error("CustomDropoutGradOpx::grow: Dropout mask shape is different from the input tensor shape!");
    }
#if OUTPUT_DROPOUT_MASK
    poplar::Tensor mask = 
#endif
    dropout.forwardOrBackwardInPlace(graph(), inOut, &seed, prog, op.getDebugStr());

    setOutTensor(CustomDropoutOp::OUT_INDEX, inOut);
#if OUTPUT_DROPOUT_MASK
    setOutTensor(CustomDropoutOp::MASK_INDEX, mask);
#endif
}

//////////////////////////////////////////////////////////////////////////
CustomDropoutGradOpx::CustomDropoutGradOpx(Op *op, popx::Devicex *devicex)
    : popx::Opx(op, devicex) {
    if (logLevel >= LogLevel::Trace)
        printf("CustomDropoutGradOpx(): this = %p, op = %p", this, op);
    verifyOp<CustomDropoutGradOp>(op, CustomGradOperators::CustomDropoutGrad);
    if (logLevel >= LogLevel::Trace) {
        CustomDropoutGradOp &dropoutGradOp = getOp<CustomDropoutGradOp>();
        printf(", fwdOp origin = %p\n", dropoutGradOp.pFwdOp);
    }
}

void CustomDropoutGradOpx::grow(poplar::program::Sequence &prog) const {
    if (logLevel >= LogLevel::Trace)
        printf("CustomDropoutGradOpx grow(): this = %p, op = %p", this, op_p);
    const CustomDropoutGradOp& dropoutGradOp = getOp<CustomDropoutGradOp>();
    if (logLevel >= LogLevel::Trace)
        printf(", fwdOp origin = %p\n", dropoutGradOp.pFwdOp);

    const poplar::Tensor& gradIn = getInTensor(CustomDropoutGradOp::D_IN_INDEX);
    const poplar::Tensor& seed = getInTensor(CustomDropoutGradOp::SEED_INDEX);

    const Dropout& dropout = DropoutCache::instance().getDropout(dropoutGradOp.getDropoutRate(), MaskTensorInfo{"DropoutMask", gradIn.shape(), graph().getTileMapping(gradIn)});
    if (dropout.getMaskTensorInfo().shape != gradIn.shape()) {
        throw error("CustomDropoutGradOpx::grow: Dropout mask shape is different from the input tensor shape!");
    }

    poplar::Tensor gradOut;
#if OUTPUT_DROPOUT_MASK
    poplar::Tensor mask;
    std::tie(gradOut, mask) = 
#else
    gradOut = 
#endif
    dropout.forwardOrBackward(graph(), gradIn, &seed, prog, dropoutGradOp.getDebugStr());

    setOutTensor(CustomDropoutGradOp::D_OUT_INDEX, gradOut);
#if OUTPUT_DROPOUT_MASK
    setOutTensor(CustomDropoutGradOp::D_DUMMY_MASK_INDEX, mask);
#endif
}

//////////////////////////////////////////////////////////////////////////
// register op
namespace {

static OpDefinition::DataTypes T  = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition
    customDropoutOpDef(
        OpDefinition::Inputs({{"data", T}}),
        OpDefinition::Outputs({{"output", T}}),
        OpDefinition::Attributes({
            {DROPOUT_RATE, OpDefinition::Attribute("*")},
            {DEBUG_STR, OpDefinition::Attribute("*")},
        })
    );

static OpCreator<CustomDropoutOp> customDropoutOpCreator(
    OpDefinitions({{CustomOperators::CustomDropout, customDropoutOpDef}}),
    [](const OpCreatorInfo &info) {
        const OperatorIdentifier &opid = info.opid;
        const Op::Settings &settings = info.settings;
        const Attributes &attr = info.attributes;

        float dropoutRate = attr.getAttribute<Attributes::Float>(DROPOUT_RATE, 0.5f);
        // If invalid probability for dropoutRate supplied, throw error.
        if (dropoutRate < float(0.) || dropoutRate >= float(1.)) {
            throw error("Custom dropout rate value {} is not valid. Please use a value in the "
                "interval [0,1)",
                dropoutRate);
        }

        std::string debugStr = attr.getAttribute<Attributes::String>("debug_str", "custom_op/CustomDropout");
        return std::unique_ptr<Op>(new CustomDropoutOp(opid, dropoutRate, settings, debugStr));
    },
    true);

static popart::popx::OpxCreator<CustomDropoutOpx> customDropoutOpxCreator(CustomOperators::CustomDropout);

static popart::popx::OpxCreator<CustomDropoutGradOpx> customDropoutGradOpxCreator(CustomGradOperators::CustomDropoutGrad);

} // namespace