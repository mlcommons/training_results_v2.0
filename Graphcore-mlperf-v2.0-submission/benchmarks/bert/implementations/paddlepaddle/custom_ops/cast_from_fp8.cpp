// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/op.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/shapeinference.hpp>

#include <popart/popx/opx.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <snap/Program.hpp>
#include <snap/Graph.hpp>

#include <popfloat/experimental/CastToGfloat.hpp>
using namespace popfloat::experimental;

using namespace popart;
using namespace popart::popx;

GfloatCast & getOrCreateCaster(poplar::Graph &graph,
                            poplar::program::Sequence &sequence,
                            poplar::Type nativeType,
                            unsigned man, unsigned exp, int bias,
                            const poplar::DebugContext &debugContext);

namespace CustomOperators {
    const popart::OperatorIdentifier CastFromFp8     = {"ai.graphcore", "CastFromFp8", 1};
    const popart::OperatorIdentifier CastFromFp8Grad = {"ai.graphcore", "CastFromFp8Grad", 1};
} // namespace CustomOperators

// CastFromFp8Op casts input tensor from FLOAT8 to FLOAT/FLOAT16, user can
// specify the number of bits for mantissa and exponent, exponent bias, which is
// different from CastOp

class CastFromFp8Op : public Op {
public:
  CastFromFp8Op(const OperatorIdentifier &_opid,
                DataType _to,
                int _nBitMantissa,
                int _nBitExponent,
                int _exponentBias,
                const Op::Settings &_settings)
    : Op(_opid, _settings), to(_to), nBitMantissa(_nBitMantissa),
      nBitExponent(_nBitExponent), exponentBias(_exponentBias) {}

  std::unique_ptr<Op> clone() const override {
      return std::make_unique<CastFromFp8Op>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup() override  {
      TensorInfo info = inInfo(getInIndex());

      info.set(to);
      outInfo(getOutIndex()) = info;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  int getNBitMantissa() { return nBitMantissa; }
  int getNBitExponent() { return nBitExponent; }
  int getExponentBias() { return exponentBias; }
  DataType getDataType() { return to; }

private:
  DataType to;
  int nBitMantissa;
  int nBitExponent;
  int exponentBias;
};

class CastFromFp8GradOp : public Op {
public:
  CastFromFp8GradOp(const OperatorIdentifier &_opid,
                    const Op::Settings &_settings)
    : Op(_opid, _settings) {}

  CastFromFp8GradOp(const CastFromFp8Op&fwdOp)
     : Op(CustomOperators::CastFromFp8Grad, fwdOp.getSettings()) {}

  std::unique_ptr<Op> clone() const final {
      return std::make_unique<CastFromFp8GradOp>(*this);
  }

  void setup() override  {
      outInfo(getOutIndex()) = inInfo(getInIndex());
  }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const std::vector<GradInOutMapper> &gradInputInfo() const final {
      static const std::vector<GradInOutMapper> inInfo = {
          {getInIndex(), CastFromFp8Op::getOutIndex(), GradOpInType::GradOut}};

    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const final {
      static const std::map<int, int> outInfo = {
          {getOutIndex(), CastFromFp8Op::getInIndex()}};

    return outInfo;
  }
};

std::vector<std::unique_ptr<Op>> CastFromFp8Op::getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<CastFromFp8GradOp>(*this));
    return upops;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::INT8};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition CastFromFp8Def({OpDefinition::Inputs({{"input", T1}}),
                                    OpDefinition::Outputs({{"output", T2}}),
                                    OpDefinition::Attributes({
                                        {"to", {"FLOAT|FLOAT16"}},
                                        {"nBitMantissa", {"INT32"}},
                                        {"nBitExponent", {"INT32"}},
                                        {"exponentBias", {"INT32"}},
                                    })});

static OpCreator<CastFromFp8Op> CastFromFp8Creator(
    OpDefinitions({
        {CustomOperators::CastFromFp8, CastFromFp8Def},
    }),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      std::string type  = oci.attributes.getAttribute<Attributes::String>("to");
      DataType dataType = dataTypeFromString(type);
      int nBitMantissa = 4;
      if (oci.attributes.hasAttribute("nBitMantissa")) {
          std::string buf  = oci.attributes.getAttribute<Attributes::String>("nBitMantissa");
          nBitMantissa = atoi(buf.c_str());
      }

      int nBitExponent = 3;
      if (oci.attributes.hasAttribute("nBitExponent")) {
          std::string buf  = oci.attributes.getAttribute<Attributes::String>("nBitExponent");
          nBitExponent = atoi(buf.c_str());
      }

      int exponentBias = 7;
      if (oci.attributes.hasAttribute("exponentBias")) {
          std::string buf  = oci.attributes.getAttribute<Attributes::String>("exponentBias");
          exponentBias = atoi(buf.c_str());
      }

      return std::unique_ptr<CastFromFp8Op>(new CastFromFp8Op(oci.opid,
                                                              dataType,
                                                              nBitMantissa,
                                                              nBitExponent,
                                                              exponentBias,
                                                              oci.settings));
    },
    true);

static popart::RegisterShapeInferenceFunction CastFromFp8ShapeInfer(
          CustomOperators::CastFromFp8,
          [](ShapeInferenceContext &ctx) {
              ctx.outInfo(0) = ctx.inInfo(0);
          });
} // namespace

///////////////////////////////////////////////////////////////////////////////
//                          CastFromFp8Opx
///////////////////////////////////////////////////////////////////////////////

class CastFromFp8Opx : public PopOpx {
public:
  CastFromFp8Opx(Op *_op, Devicex *_devicex) : PopOpx(_op, _devicex) {
      verifyOp<CastFromFp8Op>(_op);
  }

  void grow(snap::program::Sequence &prog) const final {
    CastFromFp8Op &op      = getOp<CastFromFp8Op>();
    bool enableDenorms     = true;
    bool enableInfsAndNans = false;
    int man                = op.getNBitMantissa();
    int exp                = op.getNBitExponent();
    int bias               = op.getExponentBias();

    DataType toDataType    = op.getDataType();
    poplar::Type toPoplarDataType;
    switch (toDataType) {
    case DataType::FLOAT16:
        toPoplarDataType = poplar::HALF;
        break;
    case DataType::FLOAT:
        toPoplarDataType = poplar::FLOAT;
        break;
    default:
        throw popart::error("CustomOps Error: CastFromFp8Opx::grow not support data type {}", toDataType);
    }

    auto gfCast = getOrCreateCaster(graph().getPoplarGraph(),
                                    prog.getPoplarSequence(),
                                    toPoplarDataType,
                                    man, exp, bias,
                                    debugContext("CastFromFp8/param"));
    const poplar::Tensor &in =
        getInTensor(CastFromFp8Op::getInIndex()).getPoplarTensor();
    auto out = gfCast.castGfloatToNative(graph().getPoplarGraph(),
                                        in,
                                        prog.getPoplarSequence(),
                                        debugContext("CastFromFp8/cast"));
    setOutTensor(CastFromFp8Op::getOutIndex(), snap::Tensor{out, graph()});
  }
};

class CastFromFp8GradOpx : public PopOpx {
public:
  CastFromFp8GradOpx(Op *_op, Devicex *_devicex) : PopOpx(_op, _devicex) {
      verifyOp<CastFromFp8GradOp>(_op);
  }

  void grow(snap::program::Sequence &prog) const final {
      setOutTensor(0, PopOpx::cloneNcopy(prog, getInTensor(0)));
  }
};

namespace {
OpxCreator<CastFromFp8Opx>
    CastFromFp8OpxCreator({CustomOperators::CastFromFp8});
OpxCreator<CastFromFp8GradOpx>
    CastFromFp8GradOpxCreator({CustomOperators::CastFromFp8Grad});
} // namespace
