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

protected:
  void runOnOperation() override;

private:
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

static int myrtleKernelIndex(FunctionOpInterface funcOp) {
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

static void insertMyrtleRecordCycles(FunctionOpInterface function) {
  // only time functions that we care about
  int kernelIndex = myrtleKernelIndex(function);
  if (kernelIndex == -1) {
    return;
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
    insertMyrtleRecordCycles(clone);
    table.insert(clone, std::next(function->getIterator()));
    dialect->getDmaSpecializationAttrHelper().setAttr(
        function, FlatSymbolRefAttr::get(clone));

    removeUnsupportedSpecializedOps<ComputeCoreSpecializationOpInterface>(
        function);
    removeUnsupportedSpecializedOps<DMACoreSpecializationOpInterface>(clone);

    // if (function.getName() == // delete later
    //     "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64") {
    //   function->emitWarning()
    //       << "\nAFTER SpecializeDMACode dump -- This is the rewritten kernel!!!!!\n";
    //   clone->emitWarning()
    //       << "\nAFTER SpecializeDMACode dump -- This is the cloned DMA thingy!!!!!\n";
    // }
  }
}

/*
for(mlir::Operation::reverse_iterator iter = it->rbegin(), end = it->.rend();
iter != end; ++iter){

  }

  /*
  for (BasicBlock::reverse_iterator i = BB->rbegin(), e = BB->rend(); i != e;
++i)
{
    // your code
}
  */
/*
    Value warpSzI32 = builder.create<arith::ConstantOp>(
  loc, builder.getIntegerAttr(i32Type, warpSz));
  builder.getIntegerAttr(i32Type, kernelIndex)
  Value i  = builder.create<arith::ConstantOp>(operation.getLoc(), i32Type,
  kernelIndex); Value j  = builder.create<arith::ConstantOp>(operation.getLoc(),
  i32Type, 0);
    */

// const auto& lastBlock = function.back();
// for(auto iter = lastBlock.rbegin(),){

// }
// const auto lastOp = lastBlock.back();

// *auto builder = OpBuilder::atBlockBegin(&getOperation().front());
// *auto byteShift =
//     builder.create<arith::ConstantIndexOp>(allocOp.getLoc(), offset);

// *static void build(::mlir::OpBuilder &odsBuilder,
//                    ::mlir::OperationState &odsState, ::mlir::Value i,
//                    ::mlir::Value j);
// static void build(::mlir::OpBuilder &odsBuilder,
//                   ::mlir::OperationState &odsState,
//                   ::mlir::TypeRange resultTypes, ::mlir::Value i,
//                   ::mlir::Value j);
// static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
//                   ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
//                   ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

// *return b.create<arith::ConstantIndexOp>(loc,
// attr.getValue().getSExtValue()); *Value createProduct(OpBuilder &builder,
// Location loc, ArrayRef<Value> values,
//                      Type resultType) {
//   Value one = builder.create<ConstantOp>(loc, resultType,
//                                          builder.getOneAttr(resultType));
//   ArithBuilder arithBuilder(builder, loc);
//   return std::accumulate(
//       values.begin(), values.end(), one,
//       [&arithBuilder](Value acc, Value v) { return arithBuilder.mul(acc, v);
//       });
// }

//
// auto yodel = function.getFunctionBody()//.front();
// OpBuilder builder(function.getContext());
//  auto builder = OpBuilder::atBlockBegin(&function.front());
// auto help = function.front();
// auto help = function->getIterator();//.front();
// Value id = builder.create<ComputeCoreIndexOp>(&function.front());
// auto hoodle = arith::ConstantIndexOp.build()
// Value i = builder.create<arith::ConstantIndexOp>(&function.front(),
// builder.getUI32IntegerAttr(2)); Value i  =
// builder.create<arith::ConstantIndexOp>(&function.front(), 0);
// Value i = builder.create<arith::ConstantIndexOp>()
//                                      builder.getOneAttr(resultType));
// auto myrtleOp = builder.create<MyrtleRecordCyclesOp>(
//   getOperation().getLoc(),  builder.getUI32IntegerAttr(2),
//   builder.getUI32IntegerAttr(0));
// auto builder = OpBuilder::atBlockBegin(&getOperation().front());

// clone.emitWarning() << "Here's a DMA-specific function HOODLE\n";
//
