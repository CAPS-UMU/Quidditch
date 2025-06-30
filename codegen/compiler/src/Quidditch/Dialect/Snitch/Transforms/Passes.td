#ifndef QUIDDITCH_DIALECT_SNITCH_TRANSFORMS_PASSES
#define QUIDDITCH_DIALECT_SNITCH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FormMicrokernelsPass
  : InterfacePass<"quidditch-form-microkernels", "mlir::FunctionOpInterface"> {
  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

def PromotePadsToL1Pass : Pass<"quidditch-promote-pads-to-l1"> {
  let description = [{
    Converts supported `tensor.pad` operations to `start_tensor_transfer` and
    `wait_for_tensor_copy` pairs.
  }];

  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
    "quidditch::dma::DMADialect",
  ];
}

def PromoteOperandsToL1Pass : Pass<"quidditch-promote-operands-to-l1"> {
  let description = [{
    TODO:
  }];

  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
    "quidditch::dma::DMADialect",
  ];
}

def PromoteAllocsToL1Pass : Pass<"quidditch-promote-allocs-to-l1"> {
  let description = [{
    TODO:
  }];

  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

def LowerL1AllocationsPass : InterfacePass<"quidditch-lower-l1-allocations",
  "mlir::FunctionOpInterface"> {
  let description = [{

  }];

  let options = [
    Option<"l1MemoryBytes", "l1-memory-bytes", "unsigned", "112640", "">,
    Option<"assertCompiled", "assert-compiled", "bool", "false",
           "If true, errors if the kernel does not fit into L1 meory."
           "Otherwise, removes the kernel from the output and emits a warning "
           "instead.">,
  ];
}

def LowerPipelineOpPass : Pass<"quidditch-lower-pipeline-op"> {
  let dependentDialects = [
    "mlir::scf::SCFDialect",
  ];
}

def SpecializeDMACodePass : Pass<"quidditch-specialize-dma-code",
  "mlir::ModuleOp"> {

  let description = [{
    Pass performing code specialization for DMA and compute cores while
    inserting required synchronization primitives.

    Every function originally present in the IR will be cloned and turned into
    a "dma" version.
    DMA versions have all compute operations (i.e. `mmemref.microkernel`s)
    removed while the original version has all DMA transfer operations removed.
    Barriers are inserted where data dependencies require either transfers or
    computations to have finished.

    If the timeDispatch option is enabled, calls to myrtle_record_cycles will be
    inserted at the beginning and end of the specified "dma" version.
  }];

  let options = [
    Option<"timeDispatch", "time-dispatch", "std::string", /*default=*/"",
           "Flag to time a dispatch. \'grapeFruit\' for every NsNet2 dispatch,"
           " or \'fakennMxNxK\' where M,N,K are integers">
  ];
}

def LowerForallOpPass : Pass<"quidditch-lower-forall-op"> {
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

def PipelineCopyComputePass : Pass<"quidditch-pipeline-copy-compute"> {
  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
    "quidditch::dma::DMADialect",
  ];
}

#endif
