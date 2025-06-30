#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_SPECIALIZEDMACODEPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class SpecializeDMACode
    : public quidditch::Snitch::impl::SpecializeDMACodePassBase<
          SpecializeDMACode> {
public:
  using Base::Base;
  SpecializeDMACode(const quidditch::Snitch::SpecializeDMACodePassOptions &options) {
    this->timeDispatch = options.timeDispatch;
  }

protected:
  void runOnOperation() override;

private:
std::string timeDispatch = "";
};

} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

/// Removes all operations from 'function' that implement
/// 'CoreSpecializationOpInterface' but not 'Interface'.
template <typename Interface>
static void removeUnsupportedSpecializedOps(FunctionOpInterface function) {
  function->walk([&](CoreSpecializationOpInterface operation) {
    if (isa<Interface>(*operation))
      return;

    IRRewriter rewriter(operation);
    operation.replaceWithNoop(rewriter);
  });
}

/// Inserts a barrier after every operation requiring according to
/// 'CoreSpecializationOpInterface'.
/// Note: Does not currently support barriers in divergent control flow.
static void insertBarriers(FunctionOpInterface function) {
  function->walk([](CoreSpecializationOpInterface operation) {
    if (!operation.needsSynchronization())
      return;

    OpBuilder builder(operation.getContext());
    builder.setInsertionPointAfter(operation);
    builder.create<BarrierOp>(operation->getLoc());
  });
}

static int myrtleKernelIndex(FunctionOpInterface funcOp, std::string timeDispatch) {
  // check if timeDispatch setting is set to fakennMxNxK where M, N, K, are integers
  if(timeDispatch.substr (0,6) == "fakenn"){
    std::string splittable = timeDispatch.substr(6,std::string::npos);
    std::string onlyDisp = "main$async_dispatch_0_matmul_transpose_b_"+splittable+"_f64$dma";
    if (funcOp.getName() ==
      onlyDisp) {
    return 0;
    }
    return -1;

  }
  // use the below version for grapeFruit setting (timing 5 nsnet kernels) VVV
  if(timeDispatch == "grapeFruit"){
    if (funcOp.getName() ==
      "main$async_dispatch_9_matmul_transpose_b_1x161x600_f64$dma") {
    return 0;
  }
  if (funcOp.getName() ==
      "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64$dma") {
    return 1;
  }
  if (funcOp.getName() ==
      "main$async_dispatch_7_matmul_transpose_b_1x600x400_f64$dma") {
    return 2;
  }
  if (funcOp.getName() ==
      "main$async_dispatch_8_matmul_transpose_b_1x600x600_f64$dma") {
    return 3;
  }
  if (funcOp.getName() ==
      "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64$dma") {
    return 4;
  }
  return -1;
  }
  // use the above version for grapeFruit (timing 5 nsnet kernels) ^^^
  // otherwise return failure
  return -1;
}

static bool insertMyrtleRecordCycles(FunctionOpInterface function,std::string timeDispatch) {
  // only time functions that we care about
  int kernelIndex = myrtleKernelIndex(function, timeDispatch);
  if (kernelIndex == -1) {
    return false;
  }
  // function.emitWarning("I DID find a dma func to insert timing funcs into!
  // GRAVY\n");
  int first = 0;
  // find the first operation and insert record_cycles
  for (auto &block : function) {
    for (auto &operation : block) {
      first++;
      if (first == 1) {
        OpBuilder builder(operation.getContext());
        builder.setInsertionPoint(&operation);
        Type i32Type = builder.getIntegerType(32);
        Value i = builder.create<arith::ConstantOp>(
            operation.getLoc(), builder.getIntegerAttr(i32Type, kernelIndex));
        Value j = builder.create<arith::ConstantOp>(
            operation.getLoc(), builder.getIntegerAttr(i32Type, 0));
        builder.create<MyrtleRecordCyclesOp>(operation.getLoc(), i, j);
      }
    }
  }
  // find the last operation, and insert record_cycles
  for (FunctionOpInterface::reverse_iterator it = function.rbegin(),
                                             e = function.rend();
       it != e; ++it) {
    OpBuilder builder(it->back().getContext());

    builder.setInsertionPoint(&it->back());
    Type i32Type = builder.getIntegerType(32);
    Value i = builder.create<arith::ConstantOp>(
        it->back().getLoc(), builder.getIntegerAttr(i32Type, kernelIndex));
    Value j = builder.create<arith::ConstantOp>(
        it->back().getLoc(), builder.getIntegerAttr(i32Type, 1));
    builder.create<MyrtleRecordCyclesOp>(it->back().getLoc(), i, j);

    break;
  }
  // end of function
  // function.emitWarning("I DID find a dma func to insert timing funcs into!
  // GRAVY\n");
  return true;
}

void SpecializeDMACode::runOnOperation() {
  auto *dialect = getContext().getLoadedDialect<QuidditchSnitchDialect>();
  SymbolTable table(getOperation());
  auto toSpecialize =
      llvm::to_vector(getOperation().getOps<FunctionOpInterface>());
  for (FunctionOpInterface function : toSpecialize) {
    if (function.isDeclaration())
      continue;

    insertBarriers(function);

    //   if (function.getName() == // delete later
    //     "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64") {
    //   function->emitWarning()
    //       << "\nSpecializeDMACode dump -- This is the rewritten
    //       kernel!!!!!\n";
    // }

    FunctionOpInterface clone = function.clone();
    clone.setName((clone.getName() + "$dma").str());
    // try to insert a call to our new myrtle_record_cycles function
    // only insert timing functions if time-dispatch option has been enabled
    bool inserted = false;
    if (timeDispatch != ""){
      inserted = insertMyrtleRecordCycles(clone,timeDispatch);
    }
    table.insert(clone, std::next(function->getIterator()));
    dialect->getDmaSpecializationAttrHelper().setAttr(
        function, FlatSymbolRefAttr::get(clone));

    removeUnsupportedSpecializedOps<ComputeCoreSpecializationOpInterface>(
        function);
    removeUnsupportedSpecializedOps<DMACoreSpecializationOpInterface>(clone);
    
    if (inserted) { // delete later
      // function->emitWarning()
      //     << "\nAFTER SpecializeDMACode dump -- This is the rewritten kernel!!!!!\n";
      clone->emitWarning()
          << "\nAFTER SpecializeDMACode dump -- This is the cloned DMA thingy!!!!!\n"
          << "and my time-dispatch flag is " << timeDispatch << "\n";
    }
  }
}
