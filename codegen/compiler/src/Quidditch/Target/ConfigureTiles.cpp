#include "Passes.h"

#include <sys/wait.h>
#include <unistd.h>
#include "Myrtle.h"
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

class ConfigureTiles
    : public quidditch::impl::ConfigureTilesBase<ConfigureTiles> {
public:
  using Base::Base;
  ConfigureTiles(const quidditch::ConfigureTilesOptions &options) {
    this->importTiles = options.importTiles;
    this->myrtleOut = options.myrtleOut;
    this->myrtleMode = options.myrtleMode;
    this->myrtlePath = options.myrtlePath;
    this->tbl = (quidditch::TileInfoTbl *)options.tablePointer;
  }
  std::string errs = "";

protected:
  void runOnOperation() override;

private:
  std::string importTiles = "";
  std::string myrtlePath = "";
  std::string myrtleMode = "";
  std::string myrtleOut = "";
  int acc = 0;
  quidditch::TileInfoTbl *tbl;
};
} // namespace

static LogicalResult setTranslationInfo(FunctionOpInterface funcOp) {
  return setTranslationInfo(
      funcOp,
      IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, SymbolRefAttr()));
}

static LogicalResult
setRootConfig(FunctionOpInterface funcOp, Operation *rootOp,
              quidditch::TileInfoTbl *tbl, std::string myrtlePath,
              std::string myrtleMode, std::string myrtleOut) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        std::string tileSizesPath = myrtleOut;
        std::string dispatchName = funcOp.getName().str();
        // if myrtle enabled, automatically generate tile sizes
        if (myrtlePath != "") {
          pid_t pid = fork();
          if (pid == 0) {
            char *intrepreter = (char *)"python3";
            char *pythonPath = (char *)myrtlePath.c_str();
            char *pythonArgs[] = {intrepreter,
                                  pythonPath,
                                  (char *)dispatchName.c_str(),
                                  (char *)myrtleMode.c_str(),
                                  (char *)tileSizesPath.c_str(),
                                  NULL};
            execvp(intrepreter, pythonArgs);
          }
          int status;
          wait(&status);
          if(status != 0){
            funcOp.emitWarning() << "\nMyrtle Failed with exit status "<<status<<"\n";
            return failure();
          }
          // import the tile sizes exported from myrtle
          std::string errs;
          if (quidditch::fillTileInfoTable(tbl, tileSizesPath, errs) == 0) {
            funcOp.emitWarning() << "\nImporting Tiles failed: \n"
                                 << errs << "\n";
            return failure();
          }
        }
        // if myrtle disabled, assume tiling scheme passed in with --iree-quidditch-import-tiles
        SmallVector<int64_t> workgroupTiles(3, 0);
        SmallVector<int64_t> l1Tiles(3, 0);
        SmallVector<int64_t> l1Interchange = {2, 0, 1};
        bool dualBuffer = false;
        // if table of tiling schemes is invalid, throw an error
        if (tbl == 0) {
          funcOp.emitWarning() << "\nPEPPERMINT: Table pointer is zero!!";
          return failure();
        }
        // look up the tile size, interchange, and double buffering settings
        // from table
        auto search = tbl->find(funcOp.getName().str());
        if (search == tbl->end()) {
          funcOp.emitWarning()
              << "\nPEPPERMINT: Root operation of this function "
                 "is missing tiling scheme!";
          return failure();
        }
        quidditch::TilingScheme &ts = search->second;
        if (!ts.getTiles_flat(l1Tiles)) {
          funcOp.emitWarning() << "\nPEPPERMINT: Found tiling scheme, but "
                                  "couldn't get l1 tile list!";
          return failure();
        }
        if (!ts.getOrder_flat(l1Interchange)) {
          funcOp.emitWarning() << "\nPEPPERMINT: Found tiling scheme, but "
                                  "couldn't get l1 interchange!";
          return failure();
        }
        dualBuffer = ts.getDualBuffer();
        // set lowering config according to info in table
        setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                      rootOp->getContext(), workgroupTiles,
                                      l1Tiles, l1Interchange, dualBuffer));
        return success();
      })
      .Default(success());
}

void ConfigureTiles::runOnOperation() {
  if (importTiles == "" && myrtlePath == "") {
    // skip this pass when no arguments passed
    return;
  }

  FunctionOpInterface funcOp = getOperation();

  // TODO: un-comment out check for translationInfo, instead of blindly
  // overwriting it.
  if (getTranslationInfo(funcOp))
    return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);

  if (failed(rootOp)) {
    return signalPassFailure();
  }

  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return;
  }

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp))) {
    return signalPassFailure();
  }

  // annotate linalg ops with tile sizes
  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  // only add the lowering config if one does not exist already
  if (!loweringConfig) {
    if (failed(setRootConfig(funcOp, rootOperation, tbl, myrtlePath, myrtleMode,
                             myrtleOut))) {
      funcOp.emitWarning()
          << "\nPEPPERMINT: cheesey star set root config failed\n";
      return signalPassFailure();
    }
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitWarning() << "\nPEPPERMINT: cheesey star apply patterns and "
                            "fold greedily failed\n";
    signalPassFailure();
  }
}