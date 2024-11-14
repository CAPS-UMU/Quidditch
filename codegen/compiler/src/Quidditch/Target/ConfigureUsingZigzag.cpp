#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_CONFIGUREUSINGZIGZAG
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class ConfigureUsingZigzag
    : public quidditch::impl::ConfigureUsingZigzagBase<ConfigureUsingZigzag> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

static LogicalResult setTranslationInfo(FunctionOpInterface funcOp) {
  return setTranslationInfo(
      funcOp,
      IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, SymbolRefAttr()));
}

static LogicalResult setRootConfig(FunctionOpInterface funcOp,
                                   Operation *rootOp) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        // [0]: Always one in our matvec case.

        // [1]: How many rows we are processing. Should fit in L1.
        // Should be as high as possible for subgroup distribution.
        // Should be a multiple of 8 to be further distributed to compute cores.

        // [2]: Reduction dimension. How many columns are we
        // processing at once? Cannot be distributed but has a few effects:
        // * It allows us to make [1] larger by fitting more rows into L1.
        //   This therefore also gives us more parallelism compute core wise.
        // * It makes our workgroups larger, reducing dispatch overhead and
        //   memory bandwidth (by only needing to copy loop invariant memory
        //   once + needing to copy back the result fewer times). This could
        //   come at the cost of concurrency for distributing workgroups but is
        //   only applicable once on Occamy.
        SmallVector<int64_t> workgroupTiles(3, 0);
        SmallVector<int64_t> l1Tiles(3, 0);
        SmallVector<int64_t> l1Interchange = {
            2, 0, 1}; // quidditch interchange by default
        bool dualBuffer = true;

        if (funcOp.getName() ==
            "main$async_dispatch_9_matmul_transpose_b_1x161x600_f64") {
          l1Tiles[0] = 0;
          l1Tiles[1] = 56;
          l1Tiles[2] = 100;
        }
        if (funcOp.getName() ==
            "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64") {
          l1Tiles[1] = 40;
          // TODO: Switch to 82 and true once correctness bugs are fixed.
          l1Tiles[2] = 0;
          dualBuffer = false;
          // rootOp->emitWarning() << "YODEL: found a matmulTranspose to tile!\n";
          // l1Interchange = {0, 1, 2};
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 0;
          // l1Tiles[2] = 0;
          // dualBuffer = false;
        }
        if (funcOp.getName() ==
            "main$async_dispatch_7_matmul_transpose_b_1x600x400_f64") {
          l1Tiles[0] = 0;
          l1Tiles[1] = 40;
          l1Tiles[2] = 100;
          // rootOp->emitWarning() << "YODEL: found a matmulTranspose to tile!\n";
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 30;
          // l1Tiles[2] = 40;
          // l1Interchange = {0, 1, 2}; 
        }
        if (funcOp.getName() ==
            "main$async_dispatch_8_matmul_transpose_b_1x600x600_f64") {
          l1Tiles[0] = 0;
          l1Tiles[1] = 40;
          l1Tiles[2] = 100;
          // rootOp->emitWarning() << "YODEL: found a matmulTranspose to tile!\n";
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 200;
          // l1Tiles[2] = 5;
          // l1Interchange = {0, 1, 2}; 
        }
        if (funcOp.getName() == "main$async_dispatch_1_matmul_transpose_b_"
                                "1x1200x400_f64") { // tiled by ZigZag
          // rootOp->emitWarning() << "YODEL: found a matmulTranspose to tile!\n";
          dualBuffer = false;
          l1Tiles[0] = 0;
          l1Tiles[1] = 40;
          l1Tiles[2] = 100;
          // zigzag
          // l1Interchange = {0, 1, 2};
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 240;
          // l1Tiles[2] = 40;
          // dualBuffer = false;
        }

        setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                      rootOp->getContext(), workgroupTiles,
                                      l1Tiles, l1Interchange, dualBuffer));
        return success();
      })
      .Default(success());
}

void ConfigureUsingZigzag::runOnOperation() {
  if (this->tilingSchemes.compare(
          "/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/zigzag-tiled-nsnet/"
          "zigzag-tiled-nsnet.json") != 0) {
    return;

  } else {
    // getOperation()->emitWarning()
    //     << "YODEL: found zigzag tiling scheme to process!\n";
    //     if (!this->ts.valid) {
    //   getOperation()->emitWarning()
    //       << "valid: " << this->ts.valid
    //       << " i found a ZigZag input :) tilingScheme is ["
    //       << this->tilingScheme << "]\n";
    //   this->ts.initialize(tilingScheme);
    //   getOperation()->emitWarning()
    //       << "zigzag parsed tilingScheme is [" << this->ts.str()
    //       << "] valid: " << this->ts.valid << "\n";
    // }
  }
  FunctionOpInterface funcOp = getOperation();
  if (getTranslationInfo(funcOp)) {
    eraseTranslationInfo(funcOp);
  }

  // funcOp->emitWarning()
  //     << "YODEL: inside runOperation of configureUsingZigzag\n";

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  /*
  Find the root operation for the dispatch region. The priority is:

A Linalg operation that has reduction loops.
Any other Linalg op or LinalgExt op.
An operation that implements TilingInterface.
If there are multiple operations meeting the same priority, the one closer
to the end of the function is the root op.
   */
  if (failed(rootOp)) {
    getOperation()->emitWarning() << "YODEL: couldn't find root op!\n";
    return signalPassFailure();
  }

  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    getOperation()->emitWarning()
        << "YODEL: found the root op, but it doesn't have a value!\n";
    return;
  }

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp)))
    return signalPassFailure();

  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  if (!loweringConfig) { // if the root operation has tiling settings, destroy
                         // them
    eraseLoweringConfig(rootOperation); // destroy any previous tiling settings
  }
  // TODO: instead of only thinking about rootOp, should tile ALL the linalg ops
  // inside (i think!)
  if (failed(setRootConfig(funcOp, rootOperation))) {
    return signalPassFailure();
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
