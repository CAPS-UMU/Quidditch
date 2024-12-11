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
    if (this->toRead != "") {
      importFilePathExists = true;
    }
    if (this->toAppend != "") {
      exportFilePathExists = true;
    }
    if (options.tablePointer == 0) {
      errorReadingImportFile = true;
    }
    if (errorReadingImportFile && exportFilePathExists) {
      state = exportToFile;
    } else if ((!errorReadingImportFile) && (!exportFilePathExists)) {
      state = importFromFile;

    } else {
      state = error;
    }
  }

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
  bool importFilePathExists = false;
  bool exportFilePathExists = false;
  bool errorReadingImportFile = false;
  ConfigureTilesState state = error;
};

} // namespace

void ConfigureTiles::importTileScheme(mlir::FunctionOpInterface *funcOp) {}
void ConfigureTiles::exportUntiled(mlir::FunctionOpInterface *funcOp) {}

void ConfigureTiles::runOnOperation() {
  // auto op = getOperation();
  FunctionOpInterface funcOp = getOperation();
  switch (state) {
  case exportToFile:
    exportUntiled(&funcOp);
    break;
  case importFromFile:
    importTileScheme(&funcOp);
    break;
  default:
    funcOp->emitError()
        << "ConfigureTiles: Error importing tile schemes from file "
        << this->toRead
        << ", or passed both an import and export file path (which is not "
           "permitted).\n";
    return signalPassFailure();
    break;
  }
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