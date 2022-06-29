// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "random_mask.hpp"

#include "poprand/RandomGen.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include "poplar/RandomSeed.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/exceptions.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Expr.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>
#include <cstdint>
#include <limits>

#include "rnnt_utils.hpp"

static LogLevel logLevel = getLogLevel();

/*
The code's origin is poprand/RandomGen.cpp:bernoulli
Main change is that instead of a reference tensor we pass reference tensor's layout.
This achieves the same result of determinism of the dropout needed between forward and backward pass,
but allows to not rely on popart's framework to maintain reference tensor.
*/

// flatten 2D vector of intervals to a 1D vector
static std::vector<poplar::Interval>
flatten(const std::vector<std::vector<poplar::Interval>> &intervals2D) {
  std::vector<poplar::Interval> flattenedIntervals;
  for (const auto &intervals1D : intervals2D) {
    flattenedIntervals.insert(flattenedIntervals.end(), std::begin(intervals1D),
                              std::end(intervals1D));
  }
  return flattenedIntervals;
}

static void seedTensorChecks(const poplar::Tensor *seed) {
  if (seed) {
    if (seed->rank() != 1) {
      // We could allow seed of any shape as long as it has the required number
      // of elements. For now, impose the stricter condition
      throw poputil::poplibs_error("seed tensor must have rank 1");
    }
    if (seed->numElements() != 2) {
      throw poputil::poplibs_error("seed tensor must have 2 elements");
    }
    if (seed->elementType() != poplar::UNSIGNED_INT) {
      throw poputil::poplibs_error("seed tensor must be of type UNSIGNED_INT");
    }
  }
}

void setSeed(poplar::Graph &graph, const poplar::Tensor &masterSeed,
             uint32_t seedModifier, poplar::program::Sequence &prog,
             const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(masterSeed, seedModifier));
  seedTensorChecks(&masterSeed);
  auto cs = graph.addComputeSet({di, "setMasterSeed"});
  const auto &target = graph.getTarget();
  auto numTiles = target.getNumTiles();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto v = graph.addVertex(cs, "poprand::SetSeed", {{"seed", masterSeed}});
    graph.setInitialValue(v["seedModifierUser"], seedModifier ^ 0x55555555U);
    // guarantee that even tile id 0 will have at least one bit set
    graph.setInitialValue(v["seedModifierHw"], (tile << 4) ^ 0xAAAAAAA0U);
    graph.setTileMapping(v, tile);
  }
  prog.add(poplar::program::Execute(cs, {di}));
}

// If master seed tensor is not null then read hw seeds tensor and
// program master seed
// TODO: T12982 To avoid creating vertex state for each worker within the random
// generator codelets we add the getHwSeeds and setSeed program followed by the
// setHwSeeds program. This is not efficient in both cycles and memory but
// is an expedient solution. We can revisit this if memory and performance
// becomes an issue.
static boost::optional<poplar::Tensor>
maybeSaveHwSeedsAndSetSeeds(poplar::Graph &graph, const poplar::Tensor *masterSeed,
                            uint32_t seedModifier, poplar::program::Sequence &prog,
                            const poplar::DebugNameAndId &dnai) {
  if (masterSeed) {
    auto hwSeeds = getHwSeeds(graph, prog, dnai.getPathName());
    setSeed(graph, *masterSeed, seedModifier, prog, dnai.getPathName());
    return hwSeeds;
  }
  return boost::none;
}

// Restore Hw seeds
static void maybeRestoreHwSeeds(poplar::Graph &graph,
                                const boost::optional<poplar::Tensor> &hwSeeds,
                                poplar::program::Sequence &prog, const poplar::DebugNameAndId &dnai) {
  if (hwSeeds != boost::none) {
    setHwSeeds(graph, *hwSeeds, prog, dnai.getPathName());
  }
}

poplar::Tensor genRandomMask(poplar::Graph &graph, const poplar::Tensor *masterSeed, uint32_t seedModifier,
                             poplar::Type type, const MaskTensorInfo &info, double prob,
                             poplar::program::Sequence &prog, const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(masterSeed, seedModifier, type, prob));
  seedTensorChecks(masterSeed);

  const std::string fnPrefix = "genRandomMask";

  auto out = graph.addVariable(type, info.shape, {di, fnPrefix + "/" + info.name});
  graph.setTileMapping(out, info.tileMapping);
  
  if (logLevel >= LogLevel::Verbose) {
    printf("Layout of the mask tensor %s:\n", info.name.c_str());
    std::vector<poplar::Interval> cr = out.getContiguousRegions();
    for (size_t i = 0; i < cr.size(); ++i) {
        const poplar::Interval& r = cr[i];
        printf("%lu-%lu\n", r.begin(), r.end());
    }
    printf("Tile mapping passed to the mask tensor %s:\n", info.name.c_str());
    for (size_t t = 0; t < info.tileMapping.size(); ++t) {
        for (size_t j = 0; j < info.tileMapping[t].size(); ++j) {
          const poplar::Interval& r = info.tileMapping[t][j];
          printf("%lu: %lu-%lu\n", t, r.begin(), r.end());
        }
    }
  }
 
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, {di, fnPrefix});

  auto cs = graph.addComputeSet({di, fnPrefix});
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {}, false);
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate = poputil::templateVertex("poprand::Bernoulli", type);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", poplar::concat(outFlat.slices(intervals))}});
    // The probability used by f16v4rmask/f32v2rmask is the bottom 17-bits of
    // the 2nd input operand. Hence the scaling by 2^16.
    graph.setInitialValue(v["prob"], (unsigned)(prob * 65536.0));
    graph.setTileMapping(v, tile);
  }

  prog.add(poplar::program::Execute(cs, {di}));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, {di, fnPrefix});
  di.addOutput(out);
  return out;
}