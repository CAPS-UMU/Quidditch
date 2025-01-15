/*
Simple Analytical Cost Model for Tiling in Quidditch
*/

#include "Myrtle.h"


using namespace mlir;

namespace myrtle{
   

    LogicalResult getCost(Operation *rootOp, llvm::SmallVector<int64_t>& tileSizes, llvm::SmallVector<int64_t>& out, std::string& errs){
        // only handle matrix-matrix transpose operations right now
    return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        const auto& dims = op.createFlatListOfOperandStaticDims();
        std::stringstream ss;
        ss << "Dims:[";
        for (const auto& dim : dims){
            ss << " " << dim;

        }
        ss << " ]\n";
        errs = ss.str();
        return failure();
        // if(){

        // }
          
        /*
         ::mlir::Operation::operand_range getInputs() {
    return getODSOperands(0);
  }

  ::mlir::Operation::operand_range getOutputs() {
    return getODSOperands(1);
  }
        */
      
        return success();
      })
      .Default([&](const auto& op){
        std::stringstream ss;
        ss << "\nMyrtle: Only support vector-matrix transpose operations at the moment.\n";
        errs = ss.str();
        return failure();});
    }

};