// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "get_compact_log_probs_op.hpp"
#include "dropout.hpp"
#include "rnnt_utils.hpp"

#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/popx/poplaroptionsx.hpp>
/*
#include <onnx/defs/schema.h> // NOLINT
*/
#include <poputil/TileMapping.hpp>
#include <cstdlib>
#include <fstream>
#include <array>
#include <iostream>

using namespace popart;

static LogLevel logLevel = getLogLevel();

static const char* const DEBUG_STR = "debug_str";
static const char* const DROPOUT_RATE = "dropout_rate";
static const char* const SHIFT_LABELS_BY_ONE = "shift_labels_by_one";
static const char* const SER_FACTOR = "ser_factor";
 
//////////////////////////////////////////////////////////////////////////
// Configure the output popart Tensor
GetCompactLogProbsOp::GetCompactLogProbsOp(const popart::OperatorIdentifier &_opid,
                       const popart::Op::Settings &settings_,
                       const std::string &debugStr_)
        : RandomBaseOp(_opid, OptionalDataType(), settings_)
        , debugStr(debugStr_)
{
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsOp(): this = %p\n", this);
}

std::unique_ptr<Op> GetCompactLogProbsOp::clone() const {
    return std::make_unique<GetCompactLogProbsOp>(*this);
}

void GetCompactLogProbsOp::setup() {
    if (logLevel >= LogLevel::Debug)
        printf("GetCompactLogProbsOp: setup() this = %p, name = %s\n", this, getName().c_str());
    const Shape &transShape = inInfo(TRANS_IN_INDEX).shape();
    unsigned transRank = transShape.size();
    const Shape &predShape = inInfo(PRED_IN_INDEX).shape();
    unsigned predRank = predShape.size();
    Shape joinFcWShape = inInfo(FCJOIN_W_IN_INDEX).shape();
    unsigned joinFcWRank = joinFcWShape.size();
    Shape joinFcBShape = inInfo(FCJOIN_B_IN_INDEX).shape();
    unsigned joinFcBRank = joinFcBShape.size();
    Shape labelsShape = inInfo(LABELS_IN_INDEX).shape();
    Shape labelsLenShape = inInfo(LABELS_LEN_IN_INDEX).shape();
    
    if (transRank != predRank) {
        throw error("GetCompactLogProbsOp: trans and pred tensor ranks must be the same");
    }
    unsigned joinRank = transRank;
    if (joinRank != 4) {
        throw error("GetCompactLogProbsOp: trans and pred tensors ranks must be 4");
    }
    if (labelsShape.size() != 2) {
        throw error("GetCompactLogProbsOp: labels tensor rank must be 2");
    }
    if (labelsLenShape.size() != 1) {
        throw error("GetCompactLogProbsOp: labels length tensor rank must be 1");
    }
    int64_t B = transShape[0];
    int64_t T = transShape[1];
    int64_t U = predShape[2];
    int64_t A0 = transShape[3];
    int64_t A1 = joinFcWShape[1];
    if (logLevel >= LogLevel::Debug)
        printf("GetCompactLogProbsOp: setup() B = %ld, T = %ld, U = %ld, A0 = %ld, A1 = %ld\n", B, T, U, A0, A1);

    if (transShape[2] != 1) {
        throw error("GetCompactLogProbsOp: trans tensor's dimension 2 must be 1");
    }
    if ((predShape[0] != B) || (predShape[1] != 1) || (predShape[3] != A0)) {
        throw error("GetCompactLogProbsOp: pred tensor's shape must be [B, 1, U, A0]");
    }
    if ((labelsShape[0] != B) || (labelsShape[1] != U - 1)) {
        throw error("GetCompactLogProbsOp: labels tensor shape must be [B, U - 1]");
    }
    if (labelsLenShape[0] != B) {
        throw error("GetCompactLogProbsOp: labels shape must be [B]");
    }
    if (joinFcWRank != 2) {
        throw error("GetCompactLogProbsOp: FC weight tensor rank must be 2");
    }
    if (joinFcWShape[0] != A0) {
        throw error("GetCompactLogProbsOp: FC weight tensor shape must be [A0, A1]");
    }
    if (joinFcBRank != 2) {
        throw error("GetCompactLogProbsOp: FC bias tensor rank must be 2");
    }
    if (joinFcBShape[0] != 1 || joinFcBShape[1] != A1) {
        throw error("GetCompactLogProbsOp: FC bias tensor shape must be [1, A1]");
    }

    std::vector<size_t> transShape1;
    std::vector<size_t> predShape1;
    std::vector<size_t> fcInShape;
    for (size_t i = 0; i < transShape.size(); ++i) {
        assert(transShape[i] != 0 && predShape[i] != 0);
        if (transShape[i] != predShape[i] && (transShape[i] != 1) && (predShape[i] != 1)) {
            throw error(std::string("GetCompactLogProbsOp: the dimension ") + std::to_string(i) + " of trans and pred tensors is not the same");    
        }
        transShape1.push_back(transShape[i]);
        predShape1.push_back(predShape[i]);
        fcInShape.push_back(std::max(transShape[i], predShape[i]));
    }
    std::vector<size_t> wShape1({static_cast<size_t>(joinFcWShape[0]), static_cast<size_t>(joinFcWShape[1])});

    add = std::make_unique<Add>(transShape1, predShape1, debugStr + " add");
    relu = std::make_unique<Relu>();
    try {
        fcMatmul = std::make_unique<MatMul>(fcInShape, wShape1);
    } catch (popart::error& err) {
        throw error("GetCompactLogProbsOp FC matmul: {}", err.what());
    }
    fcMatmul->setSerializationFactor(serializationFactor);

    std::vector<std::size_t> fcOutShape = {static_cast<std::size_t>(B), static_cast<std::size_t>(T), static_cast<std::size_t>(U), static_cast<std::size_t>(A1)};
    std::vector<std::size_t> fcBShape = {1u, static_cast<std::size_t>(A1)};
    fcAddBias = std::make_unique<Add>(fcOutShape, fcBShape, debugStr + " FC add bias");

    sparseLogSoftmax = std::make_unique<SparseLogSoftmax>();

    outInfo(PROBS_OUT_INDEX).set(inInfo(TRANS_IN_INDEX).dataType(), {B, T, U, 2});
#if NO_RECOMPUTE_FC_IN
    outInfo(FC_INPUT_OUT_INDEX).set(inInfo(TRANS_IN_INDEX).dataType(), {B, T, U, A0});
#endif
#if NO_RECOMPUTE_FC_OUT
    outInfo(LOGITS_OUT_INDEX).set(inInfo(TRANS_IN_INDEX).dataType(), {B, T, U, A1});
#endif
#if OUTPUT_DROPOUT_MASK
    outInfo(MASK_INDEX) = {DataType::FLOAT16, {B, T, U, A0}};
#endif
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsOp: setup() exited\n");
}

std::vector<std::unique_ptr<Op>> GetCompactLogProbsOp::getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<GetCompactLogProbsGradOp>(*this));
    return upops;
}

void GetCompactLogProbsOp::appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute(DROPOUT_RATE, dropoutRate);
    os.appendAttribute(SHIFT_LABELS_BY_ONE, shiftLabelsBy1);
    os.appendAttribute(SER_FACTOR, fcMatmul->getSerializationFactor());
}

//////////////////////////////////////////////////////////////////////////
GetCompactLogProbsGradOp::GetCompactLogProbsGradOp(const GetCompactLogProbsOp &fwdOp)
    : RandomBaseOp(CustomGradOperators::GetCompactLogProbsGrad, OptionalDataType(), fwdOp.getSettings())
    , pOp(&fwdOp)
    , fwdTransInfo(fwdOp.inInfo(GetCompactLogProbsOp::TRANS_IN_INDEX))
    , fwdPredInfo(fwdOp.inInfo(GetCompactLogProbsOp::PRED_IN_INDEX))
    , fwdJoinFcWInfo(fwdOp.inInfo(GetCompactLogProbsOp::FCJOIN_W_IN_INDEX))
    , fwdJoinFcBInfo(fwdOp.inInfo(GetCompactLogProbsOp::FCJOIN_B_IN_INDEX))
    , debugStr(fwdOp.getDebugStr())
    , add(fwdOp.getAdd())
    , relu(fwdOp.getRelu())
    , fcMatmul(fwdOp.getFcMatMul())
    , fcAddBias(fwdOp.getFcAddBias())
    , sparseLogSoftmax(fwdOp.getSparseLogSoftmax())
    , dropoutRate(fwdOp.getDropoutRate())
    , shiftLabelsBy1(fwdOp.getShiftLabelsBy1())
{
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsGradOp(): this = %p, fwdOp = %p\n", this, &fwdOp);
}

std::unique_ptr<Op> GetCompactLogProbsGradOp::clone() const {
    return std::make_unique<GetCompactLogProbsGradOp>(*this);
}

void GetCompactLogProbsGradOp::setup() {
    const Shape &transShape = fwdTransInfo.shape();
    const Shape &predShape = fwdPredInfo.shape();
    const Shape &joinFcWShape = fwdJoinFcWInfo.shape();
    const Shape &joinFcBShape = fwdJoinFcBInfo.shape();

    DataType transType = fwdTransInfo.dataType();
    DataType predType = fwdPredInfo.dataType();
    DataType joinFcWType = fwdJoinFcWInfo.dataType();
    DataType joinFcBType = fwdJoinFcBInfo.dataType();

    outInfo(D_TRANS_OUT_INDEX) = {transType, transShape};
    outInfo(D_PRED_OUT_INDEX) = {predType, predShape};
    outInfo(D_FCJOIN_W_OUT_INDEX) = {joinFcWType, joinFcWShape};
    outInfo(D_FCJOIN_B_OUT_INDEX) = {joinFcBType, joinFcBShape};
}

/* Describes the relationship of the inputs of the grad op to the
        inputs/outputs of the non-grad op */
const std::vector<GradInOutMapper> &GetCompactLogProbsGradOp::gradInputInfo() const {
    static const std::vector<GradInOutMapper> inInfo = {
    // The input of grad op at index 0 is the gradient of the output at
    // index 0 of the non-grad op
    {D_OUTPUT_IN_INDEX,         GetCompactLogProbsOp::PROBS_OUT_INDEX, GradOpInType::GradOut},// gradient of output probs
    {FCJOIN_W_IN_INDEX,         GetCompactLogProbsOp::FCJOIN_W_IN_INDEX, GradOpInType::In},   // FC weights
#if !NO_RECOMPUTE_FC_OUT
    {FCJOIN_B_IN_INDEX,         GetCompactLogProbsOp::FCJOIN_B_IN_INDEX, GradOpInType::In},   // FC biases
#endif
    {LABELS_IN_INDEX,           GetCompactLogProbsOp::LABELS_IN_INDEX, GradOpInType::In},     // labels
    {LABELS_LEN_IN_INDEX,       GetCompactLogProbsOp::LABELS_LEN_IN_INDEX, GradOpInType::In}, // labelLengths
#if NO_RECOMPUTE_FC_IN
    {FC_INPUT_OUTPUT_IN_INDEX,  GetCompactLogProbsOp::FC_INPUT_OUT_INDEX, GradOpInType::Out}, // FC output
#else
    {TRANS_IN_INDEX,            GetCompactLogProbsOp::TRANS_IN_INDEX, GradOpInType::In},      // transcription
    {PRED_IN_INDEX,             GetCompactLogProbsOp::PRED_IN_INDEX, GradOpInType::In},       // prediction
#endif
#if NO_RECOMPUTE_FC_OUT
    {LOGITS_OUTPUT_IN_INDEX,    GetCompactLogProbsOp::LOGITS_OUT_INDEX, GradOpInType::Out},   // logits
#endif
    {SEED_INDEX,                GetCompactLogProbsOp::SEED_INDEX, GradOpInType::In}           // seeds
    };
    return inInfo;
}

/* Describes the relationship of the outputs of the grad op to the
    inputs/outputs of the non-grad op */
const std::map<int, int> &GetCompactLogProbsGradOp::gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {
    {D_TRANS_OUT_INDEX, GetCompactLogProbsOp::TRANS_IN_INDEX},
    {D_PRED_OUT_INDEX, GetCompactLogProbsOp::PRED_IN_INDEX},
    {D_FCJOIN_W_OUT_INDEX, GetCompactLogProbsOp::FCJOIN_W_IN_INDEX},
    {D_FCJOIN_B_OUT_INDEX, GetCompactLogProbsOp::FCJOIN_B_IN_INDEX},
    };
    return outInfo;
}

void GetCompactLogProbsGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute(DROPOUT_RATE, dropoutRate);
    os.appendAttribute(SHIFT_LABELS_BY_ONE, shiftLabelsBy1);
    os.appendAttribute(SER_FACTOR, fcMatmul->getSerializationFactor());
}

//////////////////////////////////////////////////////////////////////////
GetCompactLogProbsOpx::GetCompactLogProbsOpx(Op *op, popx::Devicex *devicex)
    : popx::Opx(op, devicex) {
    verifyOp<GetCompactLogProbsOp>(op, CustomOperators::GetCompactLogProbs);
    graph().addCodelets("custom_ops/rnnt_loss/codelet.cpp"); // add codelets to the graph
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsOpx(): this = %p, op = %p\n", this, op);
}

void GetCompactLogProbsOpx::grow(poplar::program::Sequence &prog) const {
    if (logLevel >= LogLevel::Debug)
        printf("GetCompactLogProbsOpx: grow() this = %p, op = %p\n", this, op_p);

    poplar::Tensor trans = getInTensor(GetCompactLogProbsOp::TRANS_IN_INDEX);
    poplar::Tensor pred = getInTensor(GetCompactLogProbsOp::PRED_IN_INDEX);
    poplar::Tensor fcW = getInTensor(GetCompactLogProbsOp::FCJOIN_W_IN_INDEX);
    poplar::Tensor fcB = getInTensor(GetCompactLogProbsOp::FCJOIN_B_IN_INDEX);
    poplar::Tensor labels = getInTensor(GetCompactLogProbsOp::LABELS_IN_INDEX);
    poplar::Tensor labelLengths = getInTensor(GetCompactLogProbsOp::LABELS_LEN_IN_INDEX);
    const poplar::Tensor& seed = getInTensor(GetCompactLogProbsOp::SEED_INDEX);

    const GetCompactLogProbsOp &GetCompactLogProbs = getOp<GetCompactLogProbsOp>();
    const std::string &debugStr = GetCompactLogProbs.getDebugStr();

    poplar::Tensor addOut = GetCompactLogProbs.getAdd()->forward(graph(), trans, pred, prog, debugStr + " add");

    poplar::Tensor &reluInOut = addOut;
    GetCompactLogProbs.getRelu()->forwardInPlace(graph(), reluInOut, prog, debugStr);

    poplar::Tensor &dropoutInOut = reluInOut;
    const Dropout& dropout = DropoutCache::instance().getDropout(GetCompactLogProbs.getDropoutRate(), MaskTensorInfo{"DropoutMask", dropoutInOut.shape(), graph().getTileMapping(dropoutInOut)});
#if OUTPUT_DROPOUT_MASK
    poplar::Tensor mask = 
#endif
    dropout.forwardOrBackwardInPlace(graph(), dropoutInOut, &seed, prog, debugStr);

    poplar::Tensor &fcIn = dropoutInOut;
    poplar::Tensor fcOut = GetCompactLogProbs.getFcMatMul()->forward(graph(), fcIn, fcW, &dv_p->matmulCache, prog, debugStr);

    GetCompactLogProbs.getFcAddBias()->forwardInPlace(graph(), fcOut, fcB, prog, debugStr + " FC add bias");

    if (GetCompactLogProbs.getShiftLabelsBy1()) {
        poplar::Tensor one = graph().addConstant(labels.elementType(), {1}, 1u);
        graph().setTileMapping(one, 0);
        labels = popops::add(graph(), labels, one, prog, debugStr);
    }

    poplar::Tensor &logits = fcOut;
    poplar::Tensor out = GetCompactLogProbs.getSparseLogSoftmax()->forward(graph(), logits, labels, labelLengths, prog, debugStr);

    if (logLevel == LogLevel::Verbose) {
        std::cout << "trans: "; trans.output(std::cout);
        std::cout << "pred: "; pred.output(std::cout);
        std::cout << "fcW: "; fcW.output(std::cout);
        std::cout << "fcB: "; fcB.output(std::cout);
        std::cout << "labels: "; labels.output(std::cout);
        std::cout << "labelLengths: "; labelLengths.output(std::cout);
#if NO_RECOMPUTE_FC_IN
        std::cout << "fcIn: "; fcIn.output(std::cout);
#endif
#if NO_RECOMPUTE_FC_OUT
        std::cout << "logits: "; logits.output(std::cout);
#endif
        std::cout << "out: "; out.output(std::cout);
        std::cout << "seed: "; seed.output(std::cout);
#if OUTPUT_DROPOUT_MASK
        std::cout << "mask: "; mask.output(std::cout);
#endif
    }

    setOutTensor(GetCompactLogProbsOp::PROBS_OUT_INDEX, out);
#if NO_RECOMPUTE_FC_IN
    setOutTensor(GetCompactLogProbsOp::FC_INPUT_OUT_INDEX, fcIn);
#endif
#if NO_RECOMPUTE_FC_OUT
    setOutTensor(GetCompactLogProbsOp::LOGITS_OUT_INDEX, logits);
#endif
#if OUTPUT_DROPOUT_MASK
    setOutTensor(GetCompactLogProbsOp::MASK_INDEX, mask);
#endif
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsOpx: grow() exited\n");
}

poplar::Tensor GetCompactLogProbsOpx::createInput(InIndex index, const poplar::DebugNameAndId &dnai) const {
    std::string name = dnai.getPathName();
    if (logLevel >= LogLevel::Debug)
        printf("GetCompactLogProbsOpx: createInput() this = %p, index = %d, name = %s\n", this, index, name.c_str());
    
    GetCompactLogProbsOp &GetCompactLogProbs = getOp<GetCompactLogProbsOp>();
    if (index == GetCompactLogProbsOp::FCJOIN_W_IN_INDEX) {
        auto rhsInfo = inInfo(GetCompactLogProbsOp::FCJOIN_W_IN_INDEX);
        return GetCompactLogProbs.getFcMatMul()->createInput(graph(), popx::popType(rhsInfo), &dv_p->matmulCache, name);
    } else {
          throw error("GetCompactLogProbsOpx::createInput: Cannot create input {}", index);
    }
}

popx::InputCreatorType GetCompactLogProbsOpx::getInputCreatorType(InIndex index) const {
    return index == GetCompactLogProbsOp::FCJOIN_W_IN_INDEX ?
        popx::InputCreatorType::CanCreate :
        popx::Opx::getInputCreatorType(index);
}

bool GetCompactLogProbsOpx::createsEquiv(int, const Opx *, int) const {
    return false;
}

std::set<TensorId> GetCompactLogProbsOpx::mustExistBeforeCreate(InIndex index0) const {
    return {};
}

//////////////////////////////////////////////////////////////////////////
GetCompactLogProbsGradOpx::GetCompactLogProbsGradOpx(Op *op, popx::Devicex *devicex)
    : popx::Opx(op, devicex) {
    verifyOp<GetCompactLogProbsGradOp>(op, CustomGradOperators::GetCompactLogProbsGrad);
    if (logLevel >= LogLevel::Trace) {
        const GetCompactLogProbsGradOp &GetCompactLogProbsGrad = getOp<GetCompactLogProbsGradOp>();
        printf("GetCompactLogProbsGradOpx(): this = %p, fwdOp = %p\n", this, GetCompactLogProbsGrad.pOp);
    }
}

void GetCompactLogProbsGradOpx::grow(poplar::program::Sequence &prog) const {
    const GetCompactLogProbsGradOp &GetCompactLogProbsGrad = getOp<GetCompactLogProbsGradOp>();
    if (logLevel >= LogLevel::Debug) {
        printf("GetCompactLogProbsGradOpx: grow() this = %p, fwdOp = %p\n", this, GetCompactLogProbsGrad.pOp);
    }

    poplar::Tensor compactedGrads = getInTensor(GetCompactLogProbsGradOp::D_OUTPUT_IN_INDEX);
    poplar::Tensor fcW = getInTensor(GetCompactLogProbsGradOp::FCJOIN_W_IN_INDEX);
#if !NO_RECOMPUTE_FC_OUT
    poplar::Tensor fcB = getInTensor(GetCompactLogProbsGradOp::FCJOIN_B_IN_INDEX);
#endif
    poplar::Tensor labels = getInTensor(GetCompactLogProbsGradOp::LABELS_IN_INDEX);
    poplar::Tensor labelLengths = getInTensor(GetCompactLogProbsGradOp::LABELS_LEN_IN_INDEX);
#if NO_RECOMPUTE_FC_IN
    poplar::Tensor fcIn = getInTensor(GetCompactLogProbsGradOp::FC_INPUT_OUTPUT_IN_INDEX);
#else
    poplar::Tensor trans = getInTensor(GetCompactLogProbsGradOp::TRANS_IN_INDEX);
    poplar::Tensor pred = getInTensor(GetCompactLogProbsGradOp::PRED_IN_INDEX);
#endif
#if NO_RECOMPUTE_FC_OUT
    poplar::Tensor logits = getInTensor(GetCompactLogProbsGradOp::LOGITS_OUTPUT_IN_INDEX);
#endif
    const poplar::Tensor& seed = getInTensor(GetCompactLogProbsGradOp::SEED_INDEX);

    if (logLevel == LogLevel::Verbose) {
        std::cout << "compactedGrads: "; compactedGrads.output(std::cout);
        std::cout << "fcW: "; fcW.output(std::cout);
#if !NO_RECOMPUTE_FC_OUT
        std::cout << "fcB: "; fcB.output(std::cout);
#endif
        std::cout << "labels: "; labels.output(std::cout);
        std::cout << "labelLengths: "; labelLengths.output(std::cout);
#if NO_RECOMPUTE_FC_IN
        std::cout << "fcIn: "; fcIn.output(std::cout);
#else
        std::cout << "trans: "; trans.output(std::cout);
        std::cout << "pred: "; pred.output(std::cout);
#endif
#if NO_RECOMPUTE_FC_OUT
        std::cout << "logits: "; logits.output(std::cout);
#endif
        std::cout << "seed: "; seed.output(std::cout);
    }
    
    const std::string &debugStr = GetCompactLogProbsGrad.getDebugStr();

#if !NO_RECOMPUTE_FC_IN
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsGradOpx::grow(): recomputing add, relu\n");
    poplar::Tensor addOut = GetCompactLogProbsGrad.getAdd()->forward(graph(), trans, pred, prog, debugStr + " add");

    poplar::Tensor &reluInOut = addOut;
    GetCompactLogProbsGrad.getRelu()->forwardInPlace(graph(), reluInOut, prog, debugStr);
    
    poplar::Tensor &dropoutInOut = reluInOut;
    const Dropout& dropoutFwd = DropoutCache::instance().getDropout(GetCompactLogProbsGrad.getDropoutRate(), MaskTensorInfo{"DropoutMask", dropoutInOut.shape(), graph().getTileMapping(dropoutInOut)});
    dropoutFwd.forwardOrBackwardInPlace(graph(), dropoutInOut, &seed, prog, debugStr);

    poplar::Tensor &fcIn = dropoutInOut;
#endif
#if !NO_RECOMPUTE_FC_OUT
    if (logLevel >= LogLevel::Trace)
        printf("GetCompactLogProbsGradOpx::grow(): recomputing fc\n");
    poplar::Tensor fcOut = GetCompactLogProbsGrad.getFcMatMul()->forward(graph(), fcIn, fcW, &dv_p->matmulCache, prog, debugStr);

    GetCompactLogProbsGrad.getFcAddBias()->forwardInPlace(graph(), fcOut, fcB, prog, debugStr + " FC add bias");

    poplar::Tensor &logits = fcOut;
#endif

    if (GetCompactLogProbsGrad.getShiftLabelsBy1()) {
        poplar::Tensor one = graph().addConstant(labels.elementType(), {1}, 1u);
        graph().setTileMapping(one, 0);
        labels = popops::add(graph(), labels, one, prog, debugStr);
    }

    GetCompactLogProbsGrad.getSparseLogSoftmax()->backwardInPlace(graph(), compactedGrads, logits, labels, labelLengths,
                                                        prog, debugStr);
    poplar::Tensor &dFwdOut = logits;

    poplar::Tensor dFcIn;
    poplar::Tensor dFcW;
    poplar::Tensor dFcB;
    poplar::Tensor dFwdOutMatMul;
    GetCompactLogProbsGrad.getFcAddBias()->backward(graph(), dFwdOut, dFwdOutMatMul, dFcB, prog, debugStr + " FC add bias");

    GetCompactLogProbsGrad.getFcMatMul()->backward(graph(), dFwdOutMatMul, fcIn, fcW,
                                         dFcIn, dFcW,
                                         &dv_p->matmulCache, prog, debugStr);

    poplar::Tensor &dDropoutInOut = dFcIn;
    const Dropout& dropout = DropoutCache::instance().getDropout(GetCompactLogProbsGrad.getDropoutRate(), MaskTensorInfo{"DropoutMask", dDropoutInOut.shape(), graph().getTileMapping(dDropoutInOut)});
    dropout.forwardOrBackwardInPlace(graph(), dDropoutInOut, &seed, prog, debugStr);

    poplar::Tensor &dReluInOut = dDropoutInOut;
    poplar::Tensor &reluOut = fcIn;
    GetCompactLogProbsGrad.getRelu()->backwardInPlace(graph(), reluOut, dReluInOut, prog, debugStr);
    prog.add(poplar::program::WriteUndef(reluOut));

    poplar::Tensor dTrans;
    poplar::Tensor dPred;
    poplar::Tensor &dAddOut = dReluInOut;
    GetCompactLogProbsGrad.getAdd()->backward(graph(), dAddOut, dTrans, dPred, prog, debugStr + " add");
    prog.add(poplar::program::WriteUndef(dAddOut));

    setOutTensor(GetCompactLogProbsGradOp::D_TRANS_OUT_INDEX, dTrans);
    setOutTensor(GetCompactLogProbsGradOp::D_PRED_OUT_INDEX, dPred);
    setOutTensor(GetCompactLogProbsGradOp::D_FCJOIN_W_OUT_INDEX, dFcW);
    setOutTensor(GetCompactLogProbsGradOp::D_FCJOIN_B_OUT_INDEX, dFcB);
}

//////////////////////////////////////////////////////////////////////////
// register op
namespace {

static OpDefinition::DataTypes Ts = { DataType::FLOAT16, DataType::FLOAT };
static OpDefinition
    GetCompactLogProbsOpDef(
        OpDefinition::Inputs
        ({
            OpDefinition::Input("trans", Ts),
            OpDefinition::Input("pred", Ts),
            OpDefinition::Input("joinFcW", Ts)
        }),
        OpDefinition::Outputs
        ({
            OpDefinition::Output("Y", Ts)
        }),
        OpDefinition::Attributes
        ({
            {DEBUG_STR, OpDefinition::Attribute("*")},
            {DROPOUT_RATE, OpDefinition::Attribute("*")},
            {SHIFT_LABELS_BY_ONE, OpDefinition::Attribute("*")},
            {SER_FACTOR, OpDefinition::Attribute("*")},
        })
    );

static OpCreator<GetCompactLogProbsOp> GetCompactLogProbsOpCreator(
    OpDefinitions({{CustomOperators::GetCompactLogProbs, GetCompactLogProbsOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
            const OperatorIdentifier &opid = info.opid;
            const Op::Settings &settings = info.settings;
            const Attributes &attr = info.attributes;

            float dropoutRate = attr.getAttribute<Attributes::Float>(DROPOUT_RATE, 0.0f);
            unsigned shiftLabelsBy1 = attr.getAttribute<Attributes::Int>(SHIFT_LABELS_BY_ONE, 1U);
            unsigned serializationFactor = std::max(0U, static_cast<unsigned>(attr.getAttribute<Attributes::Int>(SER_FACTOR, 0U)));
            std::string debugStr = attr.getAttribute<Attributes::String>(DEBUG_STR, "custom_op/GetCompactLogProbs");

            auto getCompactLogProbsOp = new GetCompactLogProbsOp(opid, settings, debugStr);
            getCompactLogProbsOp->setDropoutRate(dropoutRate);
            getCompactLogProbsOp->setShiftLabelsBy1(shiftLabelsBy1);
            getCompactLogProbsOp->setSerializationFactor(serializationFactor);
            if (logLevel >= LogLevel::Debug)
                printf("GetCompactLogProbsOp: dropout_rate = %f, shift_labels_by_1 = %s, ser_factor = %u\n", dropoutRate, shiftLabelsBy1 ? "true" : "false", serializationFactor);
            return std::unique_ptr<Op>(getCompactLogProbsOp);
    },
    true);

static popart::popx::OpxCreator<GetCompactLogProbsOpx> GetCompactLogProbsOpxCreator(CustomOperators::GetCompactLogProbs);

static popart::popx::OpxCreator<GetCompactLogProbsGradOpx> GetCompactLogProbsGradOpxCreator(CustomGradOperators::GetCompactLogProbsGrad);

} // namespace