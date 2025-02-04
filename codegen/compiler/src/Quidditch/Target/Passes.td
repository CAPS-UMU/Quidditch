#ifndef QUIDDITCH_TARGET_PASSES
#define QUIDDITCH_TARGET_PASSES

include "mlir/Pass/PassBase.td"


def LinkExecutablesPass : Pass<"quidditch-link-executables", "mlir::ModuleOp"> {
  let description = [{
    Combines all `hal.executable.variant`s of the same target into a single
    `hal.executable.variant` nested within one `hal.executable`.

    Additionally performs replacement of any kernels that xDSL failed to
    compile with LLVM implementations.
  }];
}

def DisableQuidditchVariantPass : Pass<"quidditch-disable-variant",
  "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let description = [{
    Disables the `hal.executable.variant` for Quidditch if `xDSL` failed
    compilation. This is done by adding a `hal.executable.condition` returning
    false.
  }];
}

def ReluToMaxPass : Pass<"quidditch-relu-to-max">;
def PadToTilingConfigPass : Pass<"quidditch-pad-to-tiling-config">;

def ConfigureForSnitchPass
  : InterfacePass<"quidditch-configure-for-snitch",
      "mlir::FunctionOpInterface">;

// def ConfigureUsingZigzag : InterfacePass<"configure-zigzag-tile", "mlir::FunctionOpInterface"> {
//   let summary = "Annotate linalg operations with ZigZag Tiling Schemes, OR export linalg ops as ZigZag workloads.";
//   //let constructor = "createConfigureUsingZigzag()";
//   let options = [
//     Option<"tilingSchemes", "tiling-schemes", "std::string", /*default=*/"",
//            "Name of a JSON file specifying loop bounds and order for each linalg operation.">,
//     Option<"workloads", "workloads", "std::string", /*default=*/"",
//            "Name of a yaml file to which we export the zigzag workload representation of each linalg operation.">,
//   ];
// }
//IREE::HAL::ExecutableVariantOp
//let constructor = "mlir::polygeist::createConvertParallelToGPUPass2()";
// def ConfigureTiles : Pass<"quidditch-configure-tiles", "mlir::ModuleOp"> {
def ConfigureTiles : InterfacePass<"quidditch-configure-tiles", "mlir::FunctionOpInterface"> {
  let summary = "Annotate linalg operations with tile sizes";
  let description = [{
    Within each iree-defined kernel, annotate the root linalg operation with a tiling schemes (tile sizes + loop interchange).

    Alternatively, if the export option is not empty, export all iree-defined kernels 
    (that contain linalg operations) to a JSON file.
  }];
  // let constructor = "ConfigureTiles(const quidditch::ConfigureTilesOptions &options, bool valid)";
    let options = [
    Option<"importTilingSchemes", "import-tiling-schemes", "std::string", /*default=*/"",
           "Name of a JSON file specifying loop bounds and order for each linalg operation.">,
    Option<"exportUntiled", "export-untiled", "std::string", /*default=*/"",
           "Name of a JSON file to which we export untiled linalg operations.">,
    Option<"exportCosts", "export-costs", "std::string", /*default=*/"",
           "Name of a JSON file to which we export cost of each tiled linalg operation.">,
    Option<"tablePointer", "TODORemoveThisOption", "std::uintptr_t", /*default=*/"0",
           "Hacky way to prevent opening input file multiple times. Never pass a value to this option.">,
  ];
}

def ZigzagTiling : InterfacePass<"zigzag-tile", "mlir::FunctionOpInterface"> {
  let summary = "Optimize linalg operations according to a ZigZag Tiling Scheme.";
  //let constructor = "mlir::createZigzagTilingPass()";
  let options = [
    Option<"tilingScheme", "tiling-scheme", "std::string", /*default=*/"",
           "Name of a JSON file specifying loop bounds and order.">,
  ];
  //let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect", "scf::SCFDialect"];
}

def TensorTilePass : InterfacePass<"quidditch-tensor-tile",
  "mlir::FunctionOpInterface"> {
  let options = [
    Option<"tilingLevel", "tiling-level", "quidditch::TilingLevel",
           /*default=*/"quidditch::TilingLevel::L1",
           "Tiling level to tile. Supported levels are 'reduction' and 'thread'",
           [{llvm::cl::values(
              clEnumValN(quidditch::TilingLevel::L1, "l1",
                         "Tile and fuse all annotated ops to fit into L1 memory"),
              clEnumValN(quidditch::TilingLevel::Thread, "thread",
                         "Tile and fuse all annotated ops to threads")
           )}]>,
  ];
}

def RemoveTrivialLoopsPass
  : InterfacePass<"quidditch-remove-trivial-loops",
      "mlir::FunctionOpInterface"> {

}

def ConvertToLLVMPass : Pass<"quidditch-convert-to-llvm", "mlir::ModuleOp"> {
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect",
    "quidditch::SnitchDMA::SnitchDMADialect",
  ];
}

#endif
