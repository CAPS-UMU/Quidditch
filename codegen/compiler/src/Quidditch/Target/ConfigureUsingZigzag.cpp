#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ZigzagUtils.h"
#include "llvm/Support/raw_ostream.h"

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
  ConfigureUsingZigzag(const quidditch::ConfigureUsingZigzagOptions &options){
    this->tilingSchemes = options.tilingSchemes;
    this->workloads = options.workloads;
    this->tester = "HONEYBEEE";
    if(this->tilingSchemes.compare(
          "/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/zigzag-tiled-nsnet/zigzag-tiled-nsnet.json") == 0) {
            this->ts.valid = true;
          this->ts.initialize(this->tilingSchemes, this->workloads);        

    }
    else{
      this->ts.valid=false;
      this->ts.totalLoopCount=0;
    }
  }
  // ~ConfigureUsingZigzag(){
  //   // if(this->tilingSchemes.compare(
  //   //       "/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/zigzag-tiled-nsnet/zigzag-tiled-nsnet.json") == 0) {
  //   //     ts.updateWorkloads("\nI'm in the destructor!!!\t");
  //   //     ts.updateWorkloads(this->tilingSchemes);
  //   //     ts.exportWorkloadsToFile();     

  //   // }
  //   ts.updateWorkloads("\nI'm in the destructor!!!\t");
  //   ts.updateWorkloads("poodle");
  //   ts.exportWorkloadsToFile(); 
  // }
struct quidditch::TilingScheme ts;
std::string tester = "tiger!";

protected:
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConfigureUsingZigzag(const quidditch::ConfigureUsingZigzagOptions &options) {
  return std::make_unique<ConfigureUsingZigzag>(options);
}


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
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 300;
          // l1Tiles[2] = 4;
          // l1Interchange = {0, 2, 1}; 
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 30;
          // l1Tiles[2] = 40;
          // l1Interchange = {0, 2, 1};
          // dualBuffer = false;
        }
        if (funcOp.getName() ==
            "main$async_dispatch_8_matmul_transpose_b_1x600x600_f64") {
          // rootOp->emitWarning() << "YODEL: found a matmulTranspose to tile!\n";
          l1Tiles[0] = 0;
          l1Tiles[1] = 200;
          l1Tiles[2] = 5;
          l1Interchange = {0, 1, 2}; 
        }
        if (funcOp.getName() == "main$async_dispatch_1_matmul_transpose_b_"
                                "1x1200x400_f64") { // tiled by ZigZag
          // rootOp->emitWarning() << "YODEL: found a matmulTranspose to tile!\n";
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 40;
          // l1Tiles[2] = 100;
          // zigzag
          // l1Interchange = {0, 1, 2};
          // l1Tiles[0] = 0;
          // l1Tiles[1] = 240;
          // l1Tiles[2] = 40;
          // dualBuffer = false;
          l1Tiles[0] = 0;
          l1Tiles[1] = 240;
          l1Tiles[2] = 16;
          l1Interchange = {0, 2, 1}; 
        }

        setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                      rootOp->getContext(), workgroupTiles,
                                      l1Tiles, l1Interchange, dualBuffer, 89));
        return success();
      })
      .Default(success());
}

void ConfigureUsingZigzag::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
 // funcOp->emitWarning() << "BROCCOLI: looking at func... "<< funcOp.getName()<<" \n";
  funcOp->emitWarning() << "\nPOPCORN: configureUsingZigZagPass: " << funcOp.getName() << "\n";
  if((this->tilingSchemes.compare( // only run this pass for Grapefruit
          "/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/zigzag-tiled-nsnet/zigzag-tiled-nsnet.json") == 0) &&
          (this->workloads.compare("/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/zigzag-tiled-nsnet/zigzag-nsnet-workloads.yaml") == 0)){
            if(!this->ts.valid){
              //getOperation()->emitWarning() << "\nWOW some kind of tile scheme initialization error!\n" << ts.errs;
              return;
            }
            else{
              //getOperation()->emitWarning() << "\nWOW the tilescheme file name is ." << this->tilingSchemes << ". \nexport file name is ."<<this->workloads <<".\nalso tester is "<< this->tester << "\nparsed tile scheme is "<< ts.str() << "\n"; 
              //getOperation()->emitWarning() << "WOW loopCount is " << this->ts.totalLoopCount<< "\n";
              this->ts.totalLoopCount = (this->ts.totalLoopCount +1);
            }     
    std::string funcStr;
    llvm::raw_string_ostream rs = llvm::raw_string_ostream(funcStr);
    funcOp.print(rs);
   // funcOp->emitWarning() << "BROCCOLI: looking at func... "<< funcOp.getName()<<" defined as "<<rs.str() <<"\n";
  }
  else{
   // funcOp->emitWarning() << "CARROT: tilingScheme is " << this->tilingSchemes << "\n";
    return;
  }

  // FunctionOpInterface funcOp = getOperation();

  // erase any translation info created by Quidditch tiling pass
  if (getTranslationInfo(funcOp)) {
    eraseTranslationInfo(funcOp);
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  /*
  Find the root operation for the dispatch region. The priority is:

  A Linalg operation that has reduction loops.
  Any other Linalg op or LinalgExt op.
  An operation that implements TilingInterface.
  If there are multiple operations meeting the same priority, the one closer
  to the end of the function is the root op.
   */
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
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

  // erase any tiling information created by the Quidditch tiling pass 
  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  if (!loweringConfig) { // if the root operation has tiling settings, destroy
                         // them
    eraseLoweringConfig(rootOperation); // destroy any previous tiling settings
  }
  // TODO: instead of only thinking about the rootOp, 
  // should tile ALL the linalg ops inside the function
  // (Someday, remove getRootOperation() and replace with a loop)
  
  // tile the root operation
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

// notes; delete later
// ts.updateWorkloads(funcOp->getName().getIdentifier().str());
