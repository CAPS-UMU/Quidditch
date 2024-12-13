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
#include "TilingScheme.h"

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
  struct TilingScheme ts;

private:
  SmallVector<OpFoldResult>
  ZigZagTileSizeComputation(OpBuilder &builder, Operation *operation,
                            ArrayRef<ArrayRef<int64_t>> tileSizes);

  // my own hacky explorations
  LogicalResult
  tileAndFuseEach(RewriterBase &rewriter,
                  llvm::SmallDenseSet<TilingInterface> &payloadOps,
                  int tilingLevel);
  LogicalResult coconut(RewriterBase &rewriter,
                        llvm::SmallDenseSet<TilingInterface> &payloadOps,
                        int tilingLevel, FunctionOpInterface &funcOp);

  void runOnOperation() override;
};
} // namespace

void ZigzagTiling::runOnOperation() {
  if (this->tilingScheme.compare(
          "/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/"
          "snitch-cluster-only-floats-no-ssrs-dispatch_1_matmul_transpose_b_"
          "1x1200x400_f64/grapeFruit-tiling-scheme.json") == 0) {
    // if (!this->ts.valid) {
    //   getOperation()->emitWarning()
    //       << "valid: " << this->ts.valid
    //       << " i found a ZigZag input :) tilingScheme is ["
    //       << this->tilingScheme << "]\n";
    //   this->ts.initialize(tilingScheme);
    //   getOperation()->emitWarning()
    //       << "zigzag parsed tilingScheme is [" << this->ts.str()
    //       << "] valid: " << this->ts.valid << "\n";
    // }
    return;

  } else {
    // getOperation()->emitWarning()
    //     << "i should skip the ZigZag pass... tilingScheme is ["
    //     << this->tilingScheme << "]\n";
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

  // create an instance of our derived struct Pattern Rewriter.
  TrivialPatternRewriter rewriter(context);
  // Tile each Linalg Operation using a ZigZag plan
  if (failed(ZigzagTiling::coconut(rewriter, targetOps, 87, funcOp))) {
    return signalPassFailure();
  }
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

LogicalResult
ZigzagTiling::coconut(RewriterBase &rewriter,
                      llvm::SmallDenseSet<TilingInterface> &payloadOps,
                      int tilingLevel, FunctionOpInterface &funcOp) {

  for (TilingInterface tilingInterfaceOp : payloadOps) {

    auto linalgOp = cast<linalg::LinalgOp>(*tilingInterfaceOp);

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
    ArrayRef<ArrayRef<int64_t>> tileSizes = {{0}, {240}, {80}};
    const auto &ts = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // interchange vector
    ArrayRef<int64_t> interchange = {0, 2, 1};
    // do something different based on the tilingLevel parameter.

    if ((isa<linalg::MatmulTransposeBOp>(linalgOp)) &&
        (funcOp.getName() ==
         "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64")) {
      if (!this->ts.valid) {
      getOperation()->emitWarning()
          << "valid: " << this->ts.valid
          << " i found a ZigZag input :) tilingScheme is ["
          << this->tilingScheme << "]\n";
     // this->ts.initialize(tilingScheme);
      getOperation()->emitWarning()
          << "zigzag parsed tilingScheme is [" << this->ts.str()
          << "] valid: " << this->ts.valid << "\n";
    }

      linalgOp->emitWarning()
          << "case 87: I'm THE matmultransposeB "
             "operation! \n operands are... \n"
          << linalgOp->getOperands() << "\nAND current function name is\n"
          << funcOp.getName() << "\n";
      // << "\n AND my iterator types are..."
      // << linalgOp->getIteratorTypesArray()
      // <<"\n";
      // void eraseLoweringConfig(Operation *op) {
      // op->removeAttr(kConfigAttrName); }
      eraseLoweringConfig(linalgOp);
      tilingOptions.setTileSizes(ts);
      // tilingOptions.setTileSizeComputationFunction(ZigzagTiling::ZigZagTileSizeComputation);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      // tilingOptions.setInterchange(interchange); // TODO: interchange
      interchange = {0, 2, 1};
      tilingOptions.setInterchange(interchange);

    } else {
      continue;
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
          replacementOp->emitWarning() << "RADDISH WE HAVE TILED THE MATMUL AND IT'S RIGHT HERE.\n" ;//debugging
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
  } // end of each for each tilingInterfaceOp
  funcOp.emitWarning() << "RADDISH THE WHOLE FUNCTION IS HERE!";
  return success();
}

// my hacky tiling investigation
LogicalResult
ZigzagTiling::tileAndFuseEach(RewriterBase &rewriter,
                              llvm::SmallDenseSet<TilingInterface> &payloadOps,
                              int tilingLevel) {

  for (TilingInterface tilingInterfaceOp : payloadOps) {

    // auto linalgOp = dyn_cast<LinalgOp>(tilingInterfaceOp);
    // assert(linalgOp && "Tiling a linalg operation");
    auto linalgOp = cast<linalg::LinalgOp>(*tilingInterfaceOp);
    // linalgOp.getShapesToLoopsMap()

    // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
    //                         << "linalgOp's loop map is "
    //                         << linalgOp.getShapesToLoopsMap() << "\n");
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
    // ArrayRef<ArrayRef<int64_t>> tileSizes = {{4}, {4}, {4}};
    ArrayRef<ArrayRef<int64_t>> tileSizes = {{0}, {25}, {240}};
    const auto &ts = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // second level of tiling
    tileSizes = {{0}, {0}, {13}};
    const auto &ts2 = ZigzagTiling::ZigZagTileSizeComputation(
        b, tilingInterfaceOp, tileSizes);
    // interchange vector
    ArrayRef<int64_t> interchange = {0, 2, 1};
    // ArrayRef<int64_t> interchange = {2, 0, 1};
    // ArrayRef<int64_t> interchange = {2, 1, 0};
    // ArrayRef<int64_t> interchange = {2, 3, 1, 0}; // causes stack dump
    // do something different based on the tilingLevel parameter.
    switch (tilingLevel) {
    case 87:
      // std::stringstream ss;
      // linalgOp.getShapesToLoopsMap().print(ss);
      // ss << linalgOp.getShapesToLoopsMap();
      linalgOp->emitRemark()
          << "case 87: ZigZag Remark: current op's loop map is "
          << linalgOp.getLibraryCallName() << "\n";
      // SCFTilingOptions &setTileSizes(ArrayRef<OpFoldResult> ts);
      tilingOptions.setTileSizes(ts);
      // tilingOptions.setTileSizeComputationFunction(ZigzagTiling::ZigZagTileSizeComputation);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      // tilingOptions.setInterchange(interchange); // TODO: interchange
      interchange = {0, 2, 1};
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