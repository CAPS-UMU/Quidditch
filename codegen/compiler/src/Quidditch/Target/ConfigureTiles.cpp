#include "Passes.h"

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
#include <unistd.h>
#include <sys/wait.h>

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
      this->toAppend = options.exportUntiled;   
      this->toRead = options.importTilingSchemes;
      this->myrtlePath= options.exportCosts;
      this->tbl = (quidditch::TileInfoTbl *)options.tablePointer;    
  }
  std::string errs = "";

protected:
  void runOnOperation() override;
  void importTileScheme(mlir::FunctionOpInterface *funcOp);
  void exportUntiled(mlir::FunctionOpInterface *funcOp);

private:
  //std::string tester = "HONEYBEE";
  std::string toRead = "";
  std::string toAppend = "";
  std::string myrtlePath = "";
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

static LogicalResult exportRootConfig(FunctionOpInterface funcOp,
                                   Operation *rootOp,
                                   quidditch::TileInfoTbl *tbl, std::string toAppend) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        if(toAppend != ""){ 
         // funcOp.emitWarning() << "\n\nCARROT: append is "<< toAppend << "\n and appending " << funcOp.getName().str() << "\n";
          std::ofstream newFile(toAppend);
          newFile << funcOp.getName().str();
          newFile.close(); 
        }
        return success();
      })
      .Default(success());
}

static LogicalResult setRootConfig(FunctionOpInterface funcOp,
                                   Operation *rootOp,
                                   quidditch::TileInfoTbl *tbl, std::string toAppend, std::string myrtlePath) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {



      
      pid_t pid = fork(); 
      if (pid == 0)
      {
        //char* argument_list[] = {"ls", "-l", NULL}; // NULL terminated array of char* strings
        //std::cout<<"child started"<<std::endl;
        char *intrepreter= (char*) "python3"; 
        // char *pythonPath="./Pipetest.py"; 
        char *pythonPath= (char*) myrtlePath.c_str();
        char *pythonArgs[]={intrepreter,pythonPath, (char*)toAppend.c_str(),NULL};
        // char *pythonPath= (char*) targetOptions.exportCosts.c_str();
        // char *pythonArgs[]={intrepreter,pythonPath, (char*) targetOptions.exportUntiled.c_str(),NULL};
        execvp(intrepreter,pythonArgs);
        //execvp("ls", argument_list);
      }
      int status;
      wait(&status);
      std::string errs;
      quidditch::fillTileInfoTable(
          tbl, toAppend,
          errs);
      
          


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
        SmallVector<int64_t> l1Interchange = {1, 1, 1}; //{2, 0, 1};
        bool dualBuffer = false;
        SmallVector<int64_t> myrtleCost = {};

        if (tbl == 0) {
          funcOp.emitWarning() << "\nPEPPERMINT: Table pointer is zero!!";
          return failure();
          
        }        
        
        // look up the tile size, interchange, and double buffering settings from table
        auto search = tbl->find(funcOp.getName().str());
        if (search == tbl->end()) {
          funcOp.emitWarning() << "\nPEPPERMINT: Root operation of this function "
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

        std::stringstream ss("");
        ss << ts;
        funcOp.emitWarning() << "\nAFTER MYRTLE, tiling scheme is "<<ss.str()<<"\n";
       
        setLoweringConfig(rootOp,
                          quidditch::Snitch::LoweringConfigAttr::get(
                              rootOp->getContext(), workgroupTiles, l1Tiles,
                              l1Interchange, dualBuffer, myrtleCost));
        return success();
      })
      .Default(success());
}

void ConfigureTiles::runOnOperation() {
  if (toRead == "" &&
      toAppend == "") { // skip this pass when no arguments passed
    //getOperation().emitWarning() << "PEPPERMINT: No args passed, so ignoring!!";
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
    funcOp.emitWarning() << "\nPEPPERMINT: pumpkin\n";
    return signalPassFailure();
  }

  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return;
  }

  if(toRead == ""){
      // funcOp.emitWarning()
      //     << "\nPEPPERMINT: time to export this sucker!\n";
    if (failed(exportRootConfig(funcOp, rootOperation, tbl, toAppend))) {
      funcOp.emitWarning()
          << "\nExporting this tile-able dispatch failed\n";
      return signalPassFailure();
    }
    return;

  }

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp))) {
    return signalPassFailure();
  }

  // actually annotate linalg op with the tile size
  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  if (!loweringConfig) {

    if (failed(setRootConfig(funcOp, rootOperation, tbl, toAppend, myrtlePath))) {
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