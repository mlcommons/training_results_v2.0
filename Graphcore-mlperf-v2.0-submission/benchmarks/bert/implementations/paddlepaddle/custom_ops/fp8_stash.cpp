// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <popart/op.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/op/stashx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <snap/popops/ElementWise.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

using namespace popart;
using namespace popart::popx;

namespace CustomOperators {
  const popart::OperatorIdentifier Fp8Stash = {"ai.graphcore", "Fp8Stash", 1};
} // namespace CustomOperators

class Fp8StashOp : public Op {
public:
  Fp8StashOp(const OperatorIdentifier &opid_,
             int64_t stashSize_,
             const Op::Settings &settings_):
      Op(opid_, settings_), stashSize(stashSize_) {}

  std::unique_ptr<Op> clone() const override {
      return std::make_unique<Fp8StashOp>(*this);
  }

  void setup() final {
    Shape output_shape = inShape(getInIndex());
    output_shape.insert(output_shape.begin(), getStashSize());
    outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), output_shape};
  }

  static InIndex getInIndex() { return 0; }

  static OutIndex getOutIndex() { return 0; }

  int64_t getStashSize() { return stashSize; }

  TensorId getStashedTensorId() const {
      return reservedStashedPrefix() + inId(getInIndex());
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &os) const override {
      Op::appendOutlineAttributes(os);
      os.appendAttribute("stashSize", stashSize);
  }

  bool isOutlineable() const override { return false; }

private:
  int64_t stashSize;
};

class Fp8StashOpx : public PopOpx {
public:
  Fp8StashOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex){
      verifyOp<Fp8StashOp>(op);
      hStashSize = static_cast<size_t>(getOp<Fp8StashOp>().getStashSize());
  }

  void grow(snap::program::Sequence &prog) const final {
      // Create the stash size tensor.
      const auto stashSize =
          getConst(poplar::UNSIGNED_INT, {}, hStashSize, "stash_size");

      // Create the stash index tensor.
      const snap::Tensor stashIndex = graph().addVariable(
          poplar::UNSIGNED_INT, {1}, debugContext("stash_index"));
      graph().getPoplarGraph().setTileMapping(stashIndex.getPoplarTensor(), 0);
      graph().getPoplarGraph().setInitialValue(stashIndex.getPoplarTensor(),
                                            poplar::ArrayRef<uint32_t>({0}));

      // Retrieve the input tensor.
      const auto &inTensor = getInTensor(Fp8StashOp::getInIndex());

      // Create the output tensor.
      const auto outTensor =
          snap::Tensor{popops::createSliceableTensorFromSlice(
                        graph().getPoplarGraph(),
                        inTensor.expand({0}).getPoplarTensor(),
                        {0},
                        {hStashSize},
                        outId(Fp8StashOp::getOutIndex())),
                    graph()};

      // Update the stash.
      popops::dynamicUpdate(graph().getPoplarGraph(),
                        outTensor.getPoplarTensor(),
                        inTensor.expand({0}).getPoplarTensor(),
                        stashIndex.getPoplarTensor(),
                        {0},
                        {1},
                        prog.getPoplarSequence(),
                        debugContext("stash"));

      setOutTensor(Fp8StashOp::getOutIndex(), outTensor);

      // Create a "1" tensor and grow program to increment stash index by 1.
      const auto one = getConst(poplar::UNSIGNED_INT, {}, 1.0, "one");
      snap::popops::addInPlace(graph(), stashIndex, one, prog, debugContext());
      popops::remInPlace(graph().getPoplarGraph(),
                        stashIndex.getPoplarTensor(),
                        stashSize.getPoplarTensor(),
                        prog.getPoplarSequence(),
                        debugContext());
  }

private:
  size_t hStashSize;
};

namespace {
    OpxCreator<Fp8StashOpx> stashOpxCreator(CustomOperators::Fp8Stash);
} // namespace
