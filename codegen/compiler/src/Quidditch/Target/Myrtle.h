/*
Simple Analytical Cost Model for Tiling in Quidditch
*/
#include <stdio.h>
#include <unordered_map> // to store parsed tiling schemes in a hash table
#include <string.h>
#include <fstream> // to open tiling scheme file
#include <sstream>
#include <string>              // for string compare
#include "llvm/Support/JSON.h" // to parse tiling scheme
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"



namespace myrtle{
 
     mlir::LogicalResult getCost(mlir::Operation *rootOp, llvm::SmallVector<int64_t>& tileSizes, llvm::SmallVector<int64_t>& out, std::string& errs);

};