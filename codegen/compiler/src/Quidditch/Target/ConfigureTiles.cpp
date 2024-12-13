#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "TilingScheme.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_CONFIGURETILES
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

enum ConfigureTilesState { importFromFile, exportToFile, error };

class ConfigureTiles
    : public quidditch::impl::ConfigureTilesBase<ConfigureTiles> {
public:
  using Base::Base;
  ConfigureTiles(const quidditch::ConfigureTilesOptions &options) {
    // this->tilingSchemes = options.tilingSchemes;
    // this->workloads = options.workloads;
    this->tester = options.importTilingSchemes;
    this->toRead = options.importTilingSchemes;
    this->toAppend = options.exportUntiled;
    this->tbl = (quidditch::TileInfoTbl *)options.tablePointer;
  }
  std::string errs = "";
protected:
  void runOnOperation() override;
  void importTileScheme(mlir::FunctionOpInterface *funcOp);
  void exportUntiled(mlir::FunctionOpInterface *funcOp);

private:
  std::string tester = "HONEYBEE";
  std::string toRead = "";
  std::string toAppend = "";
  int acc = 0;
  quidditch::TileInfoTbl *tbl;
  
};
} // namespace

void ConfigureTiles::importTileScheme(mlir::FunctionOpInterface *funcOp) {}
void ConfigureTiles::exportUntiled(mlir::FunctionOpInterface *funcOp) {}

static LogicalResult setTranslationInfo(FunctionOpInterface funcOp) {
  return setTranslationInfo(
      funcOp,
      IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, SymbolRefAttr()));
}

static LogicalResult setRootConfig(FunctionOpInterface funcOp,
                                   Operation *rootOp, quidditch::TileInfoTbl *tbl) {
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
        SmallVector<int64_t> l1Interchange = {2, 0, 1};
        bool dualBuffer = true;
       
        if (funcOp.getName() ==
            "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64") {
          // TODO: Switch to 82 and true once correctness bugs are fixed.
          dualBuffer = false;
        }

          //     if (auto search = example.find(2); search != example.end())
    //     std::cout << "Found " << search->first << ' ' << search->second << '\n';
    // else
    //     std::cout << "Not found\n";
        // quidditch::TileInfoTbl::iterator search = tbl->find(funcOp.getName().str());
        // if(search == tbl->end()){
        //   funcOp.emitWarning() << "Root operation of this function is missing tiling scheme!";
        //   return failure();
        // }
       // struct quidditch::TilingScheme &ts = (tbl->find(std::string(funcOp.getName())))->second();
       // struct quidditch::TilingScheme& ts = (tbl->find(funcOp.getName().str()))->second();
        // else if(){
        //   funcOp.emitWarning() << "Error importing tiling scheme for this function's root operation.";
        //   return failure();
        // }

     
        setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                      rootOp->getContext(), workgroupTiles,
                                      l1Tiles, l1Interchange, dualBuffer));
        return success();
      })
      .Default(success());
}

void ConfigureTiles::runOnOperation() {
  if(toRead == "" && toAppend == ""){ // skip this pass when no arguments passed
    return;
  }

  FunctionOpInterface funcOp = getOperation();

  if(!tbl){ //export functions to json file
    return;
  }

  if (getTranslationInfo(funcOp))
    return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp))
    return signalPassFailure();
  Operation *rootOperation = rootOp.value();
  if (!rootOperation)
    return;

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp)))
    return signalPassFailure();

  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  if (!loweringConfig)
    if (failed(setRootConfig(funcOp, rootOperation, tbl)))
      return signalPassFailure();

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}

// std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
// createConfigureTiles(const quidditch::ConfigureTilesOptions &options) {
//   return std::make_unique<ConfigureTiles>(options);
// }

// std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
// createConfigureTiles(const quidditch::ConfigureTilesOptions &options, bool
// valid) {
//   return std::make_unique<ConfigureTiles>(options, valid);
// }
// std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
// createReconcileTranslationInfoPass() {
//   return std::make_unique<ReconcileTranslationInfoPass>();
// }
// mlir::ModuleOp

// std::unique_ptr<OperationPass<mlir::ModuleOp>>
// createConfigureTilesPass() {
//   return std::make_unique<ConfigureTiles>();
// }

// std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
// createConfigureTilesPass() {
//   return std::make_unique<ConfigureTiles>();
// }