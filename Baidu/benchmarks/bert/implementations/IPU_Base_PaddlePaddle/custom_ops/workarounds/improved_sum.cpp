// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <queue>
#include <stdexcept>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

#include <snap/Tensor.hpp>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <poplar/Graph.hpp>
#include <popart/graph.hpp>
#include <popart/operators.hpp>
#include <popart/region.hpp>
#include <popart/op.hpp>
#include <popart/op/sum.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier ImprovedSum = {"ai.graphcore",
                                                "ImprovedSum",
                                                1};
} // namespace CustomOperators

class ImprovedSumOp;
class ImprovedSumOpx;

// Inherit from SumOp should make this compatible with code expecting a SumOp.
class ImprovedSumOp : public popart::SumOp {
public:
  ImprovedSumOp(const popart::Op::Settings &opSettings)
      : popart::SumOp(CustomOperators::ImprovedSum, opSettings) {}
};

class ImprovedSumOpx : public popart::popx::PopOpx {
public:
  ImprovedSumOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::PopOpx(op, devicex) {
    verifyOp<ImprovedSumOp>(op, CustomOperators::ImprovedSum);
  }

  void grow(snap::program::Sequence &snap_prog) const final {
    poplar::program::Sequence &prog = snap_prog.getPoplarSequence();

    ImprovedSumOp &sumOp = getOp<ImprovedSumOp>();

    // The input tensors
    std::vector<poplar::Tensor> inputs;
    // Collect the input tensors and check if they all have the same shape
    std::vector<std::size_t> shape;
    bool needBroadcast = false;
    for (int i = 0; i < sumOp.input->n(); ++i) {
      auto t = getInTensor(i).getPoplarTensor();
      if (i == 0) {
        shape = t.shape();
      } else if (shape != t.shape()) {
        needBroadcast = true;
      }
      inputs.push_back(t);
    }

    poplar::Tensor sum;
    if (!needBroadcast) {
      for (auto i = 0; i < inputs.size(); i++) {
        inputs[i] = inputs[i].expand({0});
      }
      poplar::Tensor concatTensor = poplar::concat(inputs);
      poplar::Tensor inputTensor  = concatTensor;
      if (concatTensor.elementType() == poplar::UNSIGNED_INT) {
        inputTensor = concatTensor.reinterpret(poplar::INT);
      }

      // Compute the sum
      sum = popops::reduce(graph().getPoplarGraph(),
                           inputTensor,
                           {0},
                           {popops::Operation::ADD},
                           prog,
                           debugContext("sum"));
      if (concatTensor.elementType() == poplar::UNSIGNED_INT) {
        sum = sum.reinterpret(poplar::UNSIGNED_INT);
      }
    } else {
      // The "owner" of all expr nodes
      std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

      // The queue of expr nodes to be reduced
      std::queue<popops::expr::Expr *> expr;

      // Add the input tensors as placeholders to the expression
      for (int i = 0; i < sumOp.input->n(); ++i) {
        exprs.push_back(std::make_unique<popops::expr::PlaceHolder>(i + 1));
        expr.push(exprs.back().get());
      }

      // Build a fairly balanced binary tree
      while (expr.size() > 1) {
        auto &a = *expr.front();
        expr.pop();
        auto &b = *expr.front();
        expr.pop();

        exprs.push_back(std::make_unique<popops::expr::Add>(a, b));
        expr.push(exprs.back().get());
      }

      // Compute the sum
      sum = popops::map(graph().getPoplarGraph(),
                        *expr.front(),
                        inputs,
                        prog,
                        debugContext("sum"));
    }

    setOutTensor(ImprovedSumOp::getOutIndex(), snap::Tensor{sum, graph()});
  }

  popart::popx::InputCreatorType
  getInputCreatorType(popart::InIndex index) const final {
    // CANUNWIND if doing a series of adds.
    // Check shape doesn't change due to numpy-style broadcasting.
    // Design choice: even without broadcasting, it is possible for the
    // two inputs (of same shape) have different layout.
    // The poplar binary op can choose the layout of the output to take
    // the layout of either input.
    // However, let's layout both inputs in the same way. That way we can
    // definitely unwind through this opx, and it will also be efficient
    // when performing the op.
    if (op_p->inInfo(index) == op_p->outInfo(ImprovedSumOp::getOutIndex())) {
      return popart::popx::InputCreatorType::CanUnwind;
    } else {
      return popart::popx::InputCreatorType::Deadend;
    }
  }
  snap::Tensor unwindTensorLayout(snap::Tensor tensor,
                                  popart::InIndex inIndex,
                                  popart::OutIndex outIndex) const final {
    return tensor;
  }
  popart::view::RegMap unwindRegion(popart::InIndex,
                                    popart::OutIndex) const final {
    return [](const popart::view::Region &r) {
      return popart::view::Regions(1, r);
    };
  }
};

static popart::popx::OpxCreator<ImprovedSumOpx>
    ImprovedSumOpxCreator(CustomOperators::ImprovedSum);

class ImprovedSumPattern : public popart::PreAliasPattern {
public:
  bool matches(popart::Op *op) const override {
    // Can't use `op->isConvertibleTo<popart::SumOp>() as ImprovedSumOp inherits
    // popart::SumOp.
    return op->opid == popart::Onnx::Operators::Sum_6 ||
           op->opid == popart::Onnx::Operators::Sum_8;
  }

  std::vector<const popart::Tensor *> touches(popart::Op *) const override {
    return {};
  }

  bool apply(popart::Op *op) const override {
    auto &ir    = op->getIr();
    auto &graph = op->getGraph();

    // Collect the inputs
    auto inputs  = op->input->tensorIdMap();
    auto outputs = op->output->tensorIdMap();
    op->disconnectAllInputs();
    op->disconnectAllOutputs();

    // Create the new op
    auto improvedSum =
        graph.createConnectedOp<ImprovedSumOp>(inputs, outputs, op->settings);
    graph.topoCons->transfer(op, improvedSum);

    // Remove the old sum op
    graph.eraseOp(op->id);
    return true;
  }
};

static popart::PatternCreator<ImprovedSumPattern>
    improvedSumPatternCreator("ImprovedSumPattern", true);
