//===- ZigzagTiling.cpp - dummy hello world pass for mlir-opt ------===//
//
// I based this pass on the file
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

#include <stdio.h>
#include <string.h>
#include <fstream> // to open tiling scheme file
#include <sstream>
#include <string>              // for string compare
#include "llvm/Support/JSON.h" // to parse tiling scheme
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace quidditch {
#define GEN_PASS_DEF_ZIGZAGTILING
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace quidditch;
using namespace mlir;
using namespace mlir::iree_compiler;

#define DEBUG_TYPE "zigzag-tile"

namespace {
class ZigzagTiling : public quidditch::impl::ZigzagTilingBase<ZigzagTiling> {
public:
  using Base::Base;
  struct TilingScheme {
    bool valid = false;
    uint64_t totalLoopCount = 0;
    std::vector<std::vector<int>> bounds;
    std::vector<std::vector<int>> order;
    std::vector<std::vector<int>> finalIndices;
    TilingScheme() = default;
    void initialize(std::string filename);
    std::string str();
    friend std::stringstream &operator<<(std::stringstream &ss,
                                         const ZigzagTiling::TilingScheme &ts);

  private:
    int findSubloop(size_t i, size_t j);
    void setTotalLoopCount();
    void buildFinalIndices();
    void parseTilingScheme(StringRef fileContent);
    void parseListOfListOfInts(llvm::json::Object *obj, std::string listName,
                               std::vector<std::vector<int>> &out);
  } ts;

private:
  SmallVector<OpFoldResult>
  ZigZagTileSizeComputation(OpBuilder &builder, Operation *operation,
                            ArrayRef<ArrayRef<int64_t>> tileSizes);

  // my own hacky exploration
  LogicalResult
  tileAndFuseEach(RewriterBase &rewriter,
                  llvm::SmallDenseSet<TilingInterface> &payloadOps,
                  int tilingLevel);

  void runOnOperation() override;
};
} // namespace

void ZigzagTiling::runOnOperation() {
  if (this->tilingScheme.compare(
          "/home/hoppip/Quidditch/zigzag_tiling/zigzag-tile-scheme.json") ==
      0) {
    getOperation()->emitWarning()
        << "i found a ZigZag input :) tilingScheme is [" << this->tilingScheme
        << "]\n";
    if (!ts.valid) {
      ts.initialize(tilingScheme);
    }
    getOperation()->emitWarning()
        << "zigzag parsed tilingScheme is [" << this->ts.str() << "]\n";
    return;
  } else {
    getOperation()->emitWarning()
        << "i should skip the ZigZag pass... tilingScheme is ["
        << this->tilingScheme << "]\n";
    return;
    
  }

  llvm::SmallDenseSet<TilingInterface> targetOps;

  FunctionOpInterface funcOp = getOperation();
  // Pick out all the operations inside the current function
  // which implement a TilingInterface (linalg ops), and save them in a list.
  funcOp->walk([&](TilingInterface target) { targetOps.insert(target); });
  auto *context = &getContext(); // whose context? the function's context?
  // declare a pattern rewriter (Based on LinalgTransformOps tryApply function)
  struct TrivialPatternRewriter : public PatternRewriter {
  public:
    explicit TrivialPatternRewriter(MLIRContext *context)
        : PatternRewriter(context) {}
  };

  if (targetOps.size() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                               "] No Target Ops found inside this function!\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Target Ops now has size "
                            << targetOps.size() << "\n");
    for (const auto &op : targetOps) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] This target Op is " << op << "\n");
    }
    // create an instance of our derived struct Pattern Rewriter.
    TrivialPatternRewriter rewriter(context);
    // Tile each Linalg Operation using a ZigZag plan
    if (failed(ZigzagTiling::tileAndFuseEach(rewriter, targetOps, 87))) {
      return signalPassFailure();
    }
  }

  // LET'S DO IT A SECOND TIME!!
  // targetOps.clear();
  // funcOp = getOperation(); // I know the operation implements a function op
  //                          // interface
  // // pick out all the operations inside the current function
  // // which implement a TilingInterface, and save them in a list.
  // funcOp->walk([&](TilingInterface target) { targetOps.insert(target); });
  // context = &getContext();
  // if (targetOps.size() == 0) {
  //   LLVM_DEBUG(llvm::dbgs()
  //              << "[" DEBUG_TYPE
  //                 "] No Target Ops found inside this function after tiling!\n");
  // } else {
  //   LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Target Ops now has size "
  //                           << targetOps.size() << "\n");
  //   for (const auto &op : targetOps) {
  //     LLVM_DEBUG(llvm::dbgs()
  //                << "[" DEBUG_TYPE "] This target Op is " << op << "\n");
  //   }
  //   // create an instance of our derived struct Pattern Rewriter.
  //   TrivialPatternRewriter rewriter(context);
  //   if (failed(ZigzagTiling::tileAndFuseEach(rewriter, targetOps, 88))) {
  //     return signalPassFailure();
  //   }
  // }
}

/// This collects the set of operations to tile + fuse starting from the given
/// root |op| and walking up to its producers. Stops at operations given by
/// |exclude| which are expected to receive their own independent tiling for the
/// given level.
static llvm::SmallDenseSet<Operation *>
collectTiledAndFusedOps(Operation *op,
                        const llvm::SmallDenseSet<TilingInterface> &exclude) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || producers.contains(producer) ||
          exclude.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

// my hacky tiling investigation
LogicalResult
ZigzagTiling::tileAndFuseEach(RewriterBase &rewriter,
                              llvm::SmallDenseSet<TilingInterface> &payloadOps,
                              int tilingLevel) {
  if (tilingLevel == 87) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] inside MY tiling exploration func!\n");
  }

  std::stringstream ts_ss;
  ts_ss << ts;
  // print out what we parsed
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << "the tile scheme we have currently is...\n"
                          << ts_ss.str() << "\n");

  for (TilingInterface tilingInterfaceOp : payloadOps) {

    // auto linalgOp = dyn_cast<LinalgOp>(tilingInterfaceOp);
    // assert(linalgOp && "Tiling a linalg operation");
    auto linalgOp = cast<linalg::LinalgOp>(*tilingInterfaceOp);
    // linalgOp.getShapesToLoopsMap()

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                            << "linalgOp's loop map is "
                            << linalgOp.getShapesToLoopsMap() << "\n");
    // TODO: what does this block do? I need to find out.
    DominanceInfo dominanceInfo(tilingInterfaceOp);
    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    // repeat tiling of each loop until we are done

    rewriter.setInsertionPoint(tilingInterfaceOp);
    scf::SCFTilingOptions tilingOptions;
    OpBuilder b(tilingInterfaceOp);
    // first level of tiling
    // ArrayRef<ArrayRef<int64_t>> tileSizes = {{8}, {8}, {26}};
    ArrayRef<ArrayRef<int64_t>> tileSizes = {{4}, {4}, {4}};
    const auto &ts = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // second level of tiling
    tileSizes = {{0}, {0}, {13}};
    const auto &ts2 = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // interchange vector
    ArrayRef<int64_t> interchange = {2, 0, 1};
    // ArrayRef<int64_t> interchange = {2, 1, 0};
    // ArrayRef<int64_t> interchange = {2, 3, 1, 0}; // causes stack dump
    // do something different based on the tilingLevel parameter.
    switch (tilingLevel) {
    case 87:
      // SCFTilingOptions &setTileSizes(ArrayRef<OpFoldResult> ts);
      tilingOptions.setTileSizes(ts);
      // tilingOptions.setTileSizeComputationFunction(ZigzagTiling::ZigZagTileSizeComputation);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      // tilingOptions.setInterchange(interchange); // TODO: interchange
      tilingOptions.setInterchange(interchange);
      break;

    default:
      tileSizes = {{0}, {0}, {13}};
      const auto &ts2 = ZigzagTiling::ZigZagTileSizeComputation(
          b, tilingInterfaceOp, tileSizes);
      tilingOptions.setTileSizes(ts2);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      break;
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    // TODO: what does this block of code even do? I have to find out.
    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand) {
          Operation *owner = originalProducer.getOwner();
          bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
          bool shouldFuse = false;
          if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
            shouldFuse = !payloadOps.contains(tilingOwner);
          }
          // Do not fuse destination operands.
          shouldFuse &= !isDestinationOperand;
          return std::make_tuple(shouldFuse, yieldProducerReplacement);
        };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    // perform the tiling
    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  tileAndFuseOptions);
    if (failed(tiledResults)) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] TILING FAILED\n");
      return failure();
    }

    // TODO: what does this block really do?
    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
    // when we reach here, the entire linalg op should be tiled
  }
  return success();
}

SmallVector<OpFoldResult>
ZigzagTiling::ZigZagTileSizeComputation(OpBuilder &builder,
                                        Operation *operation,
                                        ArrayRef<ArrayRef<int64_t>> tileSizes) {
  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] Inside my zigzag tile size computation function :)\n");
  SmallVector<OpFoldResult> result;
  for (auto const &tiles : tileSizes) {
    result.push_back(builder.getIndexAttr(tiles[0]));
  }
  return result;
}

// Tiling Scheme Functions defined below

void ZigzagTiling::TilingScheme::setTotalLoopCount() {
  unsigned total = 0;
  for (const auto &bound : bounds) {
    total += (bound.size() +
              1); // for each loop getting tiled, count the extra affine loop
                  // needed to calculate the first level indexing inside a tile
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] total number of loops in tiled loop nest will be "
                   << total << " \n");
  totalLoopCount = total;
}

void ZigzagTiling::TilingScheme::buildFinalIndices() {
  // std::vector<std::vector<int>> bounds;
  // finalIndices
  for (size_t i = 0; i < bounds.size(); i++) {
    finalIndices.push_back(std::vector<int>());
    for (size_t j = 0; j < bounds[i].size(); j++) {
      size_t finalIndex = totalLoopCount - findSubloop(i, j) - 1;
      finalIndices[i].push_back(finalIndex);
    }
  }
}

int ZigzagTiling::TilingScheme::findSubloop(size_t i, size_t j) {
  for (size_t k = 0; k < order.size(); k++) {
    if (((size_t)order[k][0] == i) && ((size_t)order[k][1] == j)) {
      return k;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                             "] Error: Could not find subloop in tiling scheme "
                             "order. Returning negative index... \n");
  return -1;
}

void ZigzagTiling::TilingScheme::initialize(std::string filename) {
  // try to read file
  std::ifstream ifs(filename);
  assert(ifs.is_open() && "Tiling Scheme File exists and can be opened.");
  std::stringstream ss;
  ss << ifs.rdbuf();
  assert(ss.str().length() != 0 &&
         "Tiling Scheme file cannot have content length of 0");
  //  try to parse file contents
  parseTilingScheme(StringRef(ss.str()));
  setTotalLoopCount();
  buildFinalIndices();
  valid = true;
}

std::string ZigzagTiling::TilingScheme::str() {
  std::stringstream ts_ss;
  ts_ss << *this;
  return ts_ss.str();
}

// helpers for processing tiling scheme input
void ZigzagTiling::TilingScheme::parseListOfListOfInts(
    llvm::json::Object *obj, std::string listName,
    std::vector<std::vector<int>> &out) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) { // getAsArray returns a (const json::Array *)
    llvm::errs() << "Error: field labeled '" << listName
                 << "' does not exist \n ";
    exit(1);
  }

  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    llvm::errs() << "Error: field labeled '" << listName
                 << "' is not a JSON array \n ";
    exit(1);
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  for (const auto &Item :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!Item.getAsArray()) {
      llvm::errs() << "Error: elt of '" << listName
                   << "' is not also a JSON array \n ";
      exit(1);
    }
    std::vector<int> sublist;
    int bound;
    for (const auto &elt :
         *(Item.getAsArray())) { // loop over a json::Array type
      if (!fromJSON(elt, bound, Root)) {
        llvm::errs() << llvm::toString(Root.getError()) << "\n";
        Root.printErrorContext(elt, llvm::errs());
        exit(1);
      }
      sublist.push_back(bound);
    }
    out.push_back(sublist);
  }
}

void ZigzagTiling::TilingScheme::parseTilingScheme(StringRef fileContent) {
  llvm::Expected<llvm::json::Value> maybeParsed =
      llvm::json::parse(fileContent);
  if (!maybeParsed) {
    llvm::errs() << "Error when parsing JSON file contents: "
                 << llvm::toString(maybeParsed.takeError());
    exit(1);
  }
  // try to get the top level json object
  if (!maybeParsed->getAsObject()) {
    llvm::errs() << "Error: top-level value is not a JSON object: " << '\n';
    exit(1);
  }
  llvm::json::Object *O = maybeParsed->getAsObject();
  // try to read the two fields
  parseListOfListOfInts(O, "bounds", bounds);
  parseListOfListOfInts(O, "order", order);
}

namespace {
std::stringstream &operator<<(std::stringstream &ss,
                              const ZigzagTiling::TilingScheme &ts) {
  ss << "tiling scheme: {\nbounds: [ ";
  for (const auto &sublist : ts.bounds) {
    ss << "[ ";
    for (const auto &bound : sublist) {
      ss << " " << bound << " ";
    }
    ss << "] ";
  }
  ss << "]\n";
  ss << "finalIndices: [ ";
  for (const auto &sublist : ts.finalIndices) {
    ss << "[ ";
    for (const auto &pos : sublist) {
      ss << " " << pos << " ";
    }
    ss << "] ";
  }
  ss << "]\n}";
  ss << "order: [ ";
  for (const auto &sublist : ts.order) {
    ss << "[ ";
    for (const auto &pos : sublist) {
      ss << " " << pos << " ";
    }
    ss << "] ";
  }
  ss << "]\n}";
  return ss;
}
} // namespace