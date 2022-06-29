// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

#include <popart/op.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <snap/popops/ElementWise.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

using namespace popart;
using namespace popart::popx;

namespace CustomOperators {
  const popart::OperatorIdentifier Fp8Restore = {"ai.graphcore", "Fp8Restore", 1};
  const popart::OperatorIdentifier Fp8RestoreInplace = {"ai.graphcore", "Fp8RestoreInplace", 1};
} // namespace CustomOperators

class Fp8RestoreOp : public Op {
public:
  Fp8RestoreOp(const OperatorIdentifier &opid_,
            int64_t stashSize_,
            const Op::Settings &settings_) : 
    Op(opid_, settings_), stashSize(stashSize_) {}

  std::unique_ptr<Op> clone() const override {
      return std::make_unique<Fp8RestoreOp>(*this);
  }

  void setup() final  {
      auto stash   = input->tensor(getStashInIndex());
      auto stashOp = stash->getProducer();
      auto act     = stashOp->input->tensor(0);
      outInfo(getRestoredActOutIndex()) = act->info;
  }

  // The stash tensor from which to restore the activation tensor
  static InIndex getStashInIndex() { return 0; }

  // Returns a reference to the restored activation tensor
  static OutIndex getRestoredActOutIndex() { return 0; }
/*
  TensorId getRestoredTensorId() const  {
      auto stash   = input->tensor(getStashInIndex());
      auto stashOp = stash->getProducer();
      auto act     = stashOp->input->tensor(StashOp::getInIndex());
      return reservedRestoredPrefix() + act->id;
  }
  */

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  int64_t getStashSize() const { return stashSize; }

  void appendOutlineAttributes(OpSerialiserBase &os) const override  {
      Op::appendOutlineAttributes(os);
      os.appendAttribute("stashSize", stashSize);
  }

  bool isOutlineable() const override { return false; }

private:
  int64_t stashSize;
};

class Fp8RestoreInplaceOp : public Fp8RestoreOp {
public:
  Fp8RestoreInplaceOp(const OperatorIdentifier &opid_,
                   int64_t stashSize_,
                   const Op::Settings &settings_)
    : Fp8RestoreOp(opid_, stashSize_, settings_) {}

  std::unique_ptr<Op> clone() const override  {
      return std::make_unique<Fp8RestoreInplaceOp>(*this);
  }

  // The activation tensor to restore
  static InIndex getActToRestoreInIndex() { return 1; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Regions aliases(InIndex in, OutIndex) const final {
    if (in == getActToRestoreInIndex()) {
      return {view::Region::getFull(inShape(in), view::AccessType::Write)};
    } else {
      return {view::Region::getEmpty(inRank(in))};
    }
  }

  view::Regions modifies(InIndex index) const final {
      return aliases(index, 0);
  }

  virtual void growAliasModel(AliasModel &m) const override {
    growAliasModelMulti(m);
  }
};

template <typename Derived> class Fp8RestoreBaseOpx : public PopOpx {
public:
  Fp8RestoreBaseOpx(Op *op, Devicex *devicex) 
    : PopOpx(op, devicex) {
    verifyOp<typename Derived::OpType>(op);
  }

  void grow(snap::program::Sequence &) const = 0;

protected:

  snap::Tensor growRestore(snap::program::Sequence &prog,
                           const snap::Tensor &stash) const {
    const auto &op       = getOp<typename Derived::OpType>();
    const auto stashSize = op.getStashSize();

    // Create the stash index tensor.
    const auto stashIndex =
        graph().addVariable(poplar::UNSIGNED_INT, {1}, debugContext());
    graph().getPoplarGraph().setTileMapping(stashIndex.getPoplarTensor(), 0);
    graph().getPoplarGraph().setInitialValue(stashIndex.getPoplarTensor(),
                                            poplar::ArrayRef<uint32_t>({0}));

    // Create the stash size tensor.
    const auto stashSizeTensor =
        getConst(poplar::UNSIGNED_INT, {}, stashSize, "stash_size")
            .getPoplarTensor();

    // Grow program to take slice of stash at the stash index.
    poplar::Tensor actFromStash = popops::dynamicSlice(graph().getPoplarGraph(),
                           stash.getPoplarTensor(),
                           stashIndex.getPoplarTensor(),
                           {0},
                           {1},
                           prog.getPoplarSequence(),
                           debugContext("grow_restore_dynamic_slice"));

    // Create a "1" tensor and grow program to increment stash index by 1.
    auto one = getConst(poplar::UNSIGNED_INT, {}, 1.0, "one");
    snap::popops::addInPlace(graph(), stashIndex, one, prog, debugContext());
    popops::remInPlace(graph().getPoplarGraph(),
                      stashIndex.getPoplarTensor(),
                      stashSizeTensor,
                      prog.getPoplarSequence(),
                      debugContext());

    return snap::Tensor{actFromStash.squeeze({0}), graph()};
  }
};

class Fp8RestoreOpx final : public Fp8RestoreBaseOpx<Fp8RestoreOpx> {
public:
  Fp8RestoreOpx(Op *op_, Devicex *devicex_) 
    : Fp8RestoreBaseOpx(op_, devicex_) {}

  void grow(snap::program::Sequence &prog) const final  {
    auto stash = getInTensor(Fp8RestoreOp::getStashInIndex());

    auto actFromStash = growRestore(prog, stash);

    setOutTensor(Fp8RestoreOp::getRestoredActOutIndex(), actFromStash);
  }

  using OpType = Fp8RestoreOp;
};

class Fp8RestoreInplaceOpx final : public Fp8RestoreBaseOpx<Fp8RestoreInplaceOpx> {
public:
  Fp8RestoreInplaceOpx(Op *op_, Devicex *devicex_)
    : Fp8RestoreBaseOpx(op_, devicex_) {}

  void grow(snap::program::Sequence &prog) const final  {
    auto actToRestore = getInTensor(RestoreInplaceOp::getActToRestoreInIndex());
    auto stash        = getInTensor(RestoreInplaceOp::getStashInIndex());

    const auto actFromStash = growRestore(prog, stash);

    prog.add(
        snap::program::Copy(actFromStash, actToRestore, false, debugContext()));
    setOutTensor(RestoreInplaceOp::getRestoredActOutIndex(), actToRestore);
  }

  using OpType = Fp8RestoreInplaceOp;
};

namespace {
OpxCreator<Fp8RestoreOpx> fp8RestoreOpxCreator(CustomOperators::Fp8Restore);
OpxCreator<Fp8RestoreInplaceOpx> fp8RestoreInplaceOpxCreator(CustomOperators::Fp8RestoreInplace);
} // namespace
