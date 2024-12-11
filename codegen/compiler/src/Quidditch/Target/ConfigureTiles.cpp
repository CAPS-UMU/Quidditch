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
#define GEN_PASS_DEF_CONFIGURETILES
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

namespace {
class ConfigureTiles : public quidditch::impl::ConfigureTilesBase<ConfigureTiles> {
public:
  using Base::Base;
    ConfigureTiles(const quidditch::ConfigureTilesOptions &options){
    // this->tilingSchemes = options.tilingSchemes;
    // this->workloads = options.workloads;
    this->tester = options.importTilingSchemes;

  }
  std::string tester = "HONEYBEE";
  int acc = 0;

protected:
  void runOnOperation() override;
};


// namespace quidditch {
// #define GEN_PASS_DEF_CONFIGURETILESPASS
// #include "Quidditch/Target/Passes.h.inc"
// } // namespace quidditch

// using namespace mlir;
// using namespace mlir::iree_compiler;
// using namespace quidditch::Snitch;

// namespace {
// class ConfigureTiles
//     : public quidditch::impl::ConfigureTilesPassBase<ConfigureTiles> {
// public:
//   using Base::Base;

// protected:
//   void runOnOperation() override;
// };
 } // namespace

void ConfigureTiles::runOnOperation() {
  //auto op = getOperation();
  // if(this->tester != "grapefruit.json"){
  //   return;

  // }
  // ModuleOp moduleOp = getOperation();
  //   std::string funcStr;
  //   llvm::raw_string_ostream rs = llvm::raw_string_ostream(funcStr);
  //   moduleOp.print(rs);
  //  // moduleOp->emitWarning() << "POPCORN: "<< this->tester << " before walk, acc is ."<< acc << "."<< rs.str()<<"\n";
  // this->acc ++;
  //   moduleOp.walk([&](FunctionOpInterface funcOp){
  //     this->acc ++;
  //     funcOp->emitWarning() << "\nPOPCORN: " << funcOp.getName() << "\t." << this->tester << ".\n";
  //     // funcOp->emitWarning() << "\nPOPCORN: tester is ."<< this->tester << ". and looking at func... "<< funcOp.getName()<<"\n";
  //     //this->tester = funcOp.getName();
  //   });
   //moduleOp->emitWarning() << "POPCORN: "<< this->tester << " after walk, acc is ."<< acc << ".\n";
    // emitWarning(moduleOp.getLoc(), "Replaced ")
    //     << xDSLFunctionsReplaced << " out of " << xDSLFunctionsEncountered
    //     << " kernels with LLVM implementations as they failed to compile";
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConfigureTiles(const quidditch::ConfigureTilesOptions &options) {
  return std::make_unique<ConfigureTiles>(options);
}
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