// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CastFromFp8Op casts input tensor from FLOAT8 to FLOAT/FLOAT16, user can
// specify the number of bits for mantissa and exponent, exponent bias, which is
// different from CastOp

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
#include <popfloat/experimental/codelets.hpp>
using namespace popfloat::experimental;

using namespace popart;
using namespace popart::popx;

namespace CustomOperators {
    const popart::OperatorIdentifier CastToFp8     = {"ai.graphcore", "CastToFp8",     1};
    const popart::OperatorIdentifier CastToFp8Grad = {"ai.graphcore", "CastToFp8Grad", 1};
} // namespace CustomOperators

SpecType poplarTypeToSpecType(poplar::Type dType) {
  if (dType == poplar::FLOAT) {
    return SpecType::FP32;
  } else if (dType == poplar::HALF) {
    return SpecType::FP16;
  } else if (dType == poplar::INT) {
    return SpecType::INT32;
  } else if (dType == poplar::SHORT) {
    return SpecType::INT16;
  } else if (dType == poplar::SIGNED_CHAR) {
    return SpecType::INT8;
  }
  throw popart::error("BERT custom op: format {} not supported in fp8 emulator", dType);
}


GfloatCast createCaster(poplar::Graph &graph,
                       poplar::program::Sequence &sequence,
                       poplar::Type nativeType,
                       unsigned man, unsigned exp, int bias,
                       const poplar::DebugContext &debugContext) {
  addCodelets(graph);
  SpecType specCalculationType = poplarTypeToSpecType(nativeType);
  bool enableDenorms           = true;
  bool enableInfsAndNans       = false;
  bool enableNanoo             = false;
  bool enableNanooMode         = enableNanoo && enableInfsAndNans && (exp > 0);
  int numberSRBits             = 23;
  auto formatConfig = GfloatCast::FormatConfig(
      man, exp, bias, enableDenorms, enableInfsAndNans, specCalculationType);
  auto roundConfig  = GfloatCast::RoundConfig(
    popfloat::experimental::RoundType::RN, numberSRBits, formatConfig.getCalculationType());
  auto caster = GfloatCast(
    formatConfig, roundConfig, enableNanooMode, SpecType::INT8, specCalculationType);
  caster.createCastOpParamsTensor(graph, sequence, debugContext);
  return caster;
}

static std::unordered_map<poplar::Graph*, std::unordered_map<std::string, GfloatCast>> casterMap;

GfloatCast & getOrCreateCaster(poplar::Graph &graph,
                            poplar::program::Sequence &sequence,
                            poplar::Type nativeType,
                            unsigned man, unsigned exp, int bias,
                            const poplar::DebugContext &debugContext) {
  // construct unique key for the given configuration
  std::string typeStr = nativeType.toString();
  typeStr.erase(std::remove(typeStr.begin(), typeStr.end(), ' '), typeStr.end());
  std::ostringstream oss;
  oss << typeStr << "_m" << man << "_e" << exp << "_b" << bias;
  std::string key = oss.str();
  // if there is no submap for the graph, create one with a new caster in it and insert
  if (casterMap.find(&graph) == casterMap.end()) {
    std::unordered_map<std::string, GfloatCast> submap;
    GfloatCast caster = createCaster(graph, sequence, nativeType, man, exp, bias, debugContext);
    submap.insert(std::make_pair(key, caster));
    casterMap.insert(std::make_pair(&graph, submap));
  } else {
    // if the submap for the graph does not have a matching key, create a new caster and insert
    if (casterMap.at(&graph).find(key) == casterMap.at(&graph).end()) {
      GfloatCast caster = createCaster(graph, sequence, nativeType, man, exp, bias, debugContext);
      casterMap.at(&graph).insert(std::make_pair(key, caster));
    }
  }
  return casterMap.at(&graph).at(key);
}

// CastToFp8Op casts input tensor from FLOAT/FLOAT16 to FLOAT8, user can specify
// the number of bits for mantissa and exponent, exponent bias, which is
// different from CastOp
class CastToFp8Op : public Op {
public:
  CastToFp8Op(const OperatorIdentifier &_opid,
              int _nBitMantissa,
              int _nBitExponent,
              int _exponentBias,
              const Op::Settings &_settings) 
    : Op(_opid, _settings), nBitMantissa(_nBitMantissa),
      nBitExponent(_nBitExponent), exponentBias(_exponentBias) {}

  std::unique_ptr<Op> clone() const override {
      return std::make_unique<CastToFp8Op>(*this);
  }

  void setup() override {
      TensorInfo info = inInfo(getInIndex());

      info.set(DataType::INT8);
      outInfo(getOutIndex()) = info;
  }

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  int getNBitMantissa() { return nBitMantissa; }
  int getNBitExponent() { return nBitExponent; }
  int getExponentBias() { return exponentBias; }

private:
  int nBitMantissa;
  int nBitExponent;
  int exponentBias;
};

class CastToFp8GradOp : public Op {
public:
  CastToFp8GradOp(const OperatorIdentifier &_opid,
                  const Op::Settings &_settings)
    : Op(_opid, _settings) {}

  CastToFp8GradOp(const CastToFp8Op&fwdOp) 
     : Op(CustomOperators::CastToFp8Grad, fwdOp.getSettings()) {}

  std::unique_ptr<Op> clone() const final {
      return std::make_unique<CastToFp8GradOp>(*this);
  }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override  {
      outInfo(getOutIndex()) = inInfo(getInIndex());
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const std::vector<GradInOutMapper> &gradInputInfo() const final {
      static const std::vector<GradInOutMapper> inInfo = {
          {getInIndex(), CastToFp8Op::getOutIndex(), GradOpInType::GradOut}};

    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const final {
      static const std::map<int, int> outInfo = {
          {getOutIndex(), CastToFp8Op::getInIndex()}};

    return outInfo;
  }
};

std::vector<std::unique_ptr<Op>> CastToFp8Op::getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<CastToFp8GradOp>(*this));
    return upops;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T2 = {DataType::INT8};

static OpDefinition CastToFp8Def({OpDefinition::Inputs({{"input", T1}}),
                                  OpDefinition::Outputs({{"output", T2}}),
                                  OpDefinition::Attributes({
                                      {"nBitMantissa", {"INT32"}},
                                      {"nBitExponent", {"INT32"}},
                                      {"exponentBias", {"INT32"}},
                                  })});

static OpCreator<CastToFp8Op> CastToFp8Creator(
    OpDefinitions({
        {CustomOperators::CastToFp8, CastToFp8Def},
    }),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
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

      return std::unique_ptr<CastToFp8Op>(new CastToFp8Op(
          oci.opid, nBitMantissa, nBitExponent, exponentBias, oci.settings));
    },
    true);

static popart::RegisterShapeInferenceFunction CastToFp8ShapeInfer(
          CustomOperators::CastToFp8,
          [](ShapeInferenceContext &ctx) {
              ctx.outInfo(0) = ctx.inInfo(0);
          });      
} // namespace

///////////////////////////////////////////////////////////////////////////////
//                          CastToFp8Opx
///////////////////////////////////////////////////////////////////////////////

class CastToFp8Opx : public PopOpx {
public:
  CastToFp8Opx(Op *_op, Devicex *_devicex) : PopOpx(_op, _devicex) {
      verifyOp<CastToFp8Op>(_op);
  }

  void grow(snap::program::Sequence &prog) const final {
    popfloat::experimental::addCodelets(graph().getPoplarGraph());

    CastToFp8Op &op        = getOp<CastToFp8Op>();
    bool enableDenorms     = true;
    bool enableInfsAndNans = false;
    int man                = op.getNBitMantissa();
    int exp                = op.getNBitExponent();
    int bias               = op.getExponentBias();

    const poplar::Tensor &in =
        getInTensor(CastToFp8Op::getInIndex()).getPoplarTensor();

    poplar::Type fromDataType = in.elementType();

    auto gfCast = getOrCreateCaster(graph().getPoplarGraph(),
                                    prog.getPoplarSequence(),
                                    fromDataType,
                                    man, exp, bias,
                                    debugContext("CastToFp8/param"));

    auto out = gfCast.castNativeToGfloat(graph().getPoplarGraph(),
                                        in,
                                        prog.getPoplarSequence(),
                                        debugContext("CastToFp8/cast"));

    setOutTensor(CastToFp8Op::getOutIndex(), snap::Tensor{out, graph()});
  }
};

class CastToFp8GradOpx : public PopOpx {
public:
  CastToFp8GradOpx(Op *_op, Devicex *_devicex) : PopOpx(_op, _devicex) {
      verifyOp<CastToFp8GradOp>(_op);
  }

  void grow(snap::program::Sequence &prog) const final {
      setOutTensor(0, PopOpx::cloneNcopy(prog, getInTensor(0)));
  }
};

namespace {
OpxCreator<CastToFp8Opx>
    CastToFp8OpxCreator({CustomOperators::CastToFp8});
OpxCreator<CastToFp8GradOpx>
    CastToFp8GradOpxCreator({CustomOperators::CastToFp8Grad});
} // namespace