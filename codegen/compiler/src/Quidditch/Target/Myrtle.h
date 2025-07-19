/*
Simple Analytical Cost Model for Tiling in Quidditch
*/
#include <stdio.h>
#include <string.h>
#include <fstream> // to open tiling scheme file
#include <sstream>
#include <string>        // for string compare
#include <unordered_map> // to store parsed tiling schemes in a hash table
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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

namespace myrtle {

enum MEMSPACE { L3, L1 };

// Result of tiling an operand in dimension dim
typedef struct LoopTileInfo {
  llvm::SmallVector<int64_t> tileShape = {};
  unsigned int tileCount = 0;
  unsigned int dim = 0;
  MEMSPACE mem = L3;
  LoopTileInfo(unsigned int dimension, unsigned int count, MEMSPACE memory,
               llvm::ArrayRef<int64_t> shape)
      : tileShape({}), tileCount(count), dim(dimension), mem(memory) {
    for (const auto &sz : shape) {
      tileShape.push_back(sz);
    }
  }
  // TODO: pick one: small vector or ArrayRef
  LoopTileInfo(unsigned int dimension, unsigned int count, MEMSPACE memory,
               llvm::SmallVector<int64_t> shape)
      : tileShape({}), tileCount(count), dim(dimension), mem(memory) {
    for (const auto &sz : shape) {
      tileShape.push_back(sz);
    }
  }
} LoopTileInfo;

// Given that tiling is expressed as a nested loop,
// what are the tile sizes and tile counts
// for each operand at each loop nest depth?
typedef struct OperandTileInfo {
  mlir::OpOperand *operand = 0;
  llvm::SmallVector<LoopTileInfo> loops = {};
  int64_t cycles = 0;
  int64_t L3toL1reads = 0;
  int64_t L1toRFreads = 0;
  OperandTileInfo(mlir::OpOperand *oper) : operand(oper) {}
  void tileLoopByLoop(mlir::linalg::LinalgOp &op,
                      llvm::ArrayRef<int64_t> &tileSizes,
                      llvm::ArrayRef<int64_t> &interchange,
                      MEMSPACE innermostLoopMemspace);
  void tileOneLoopMore(mlir::linalg::LinalgOp &op,
                      int64_t loopBound,
                      int64_t dim,
                      MEMSPACE memspace);
  void computeValues(std::string& errs);

} OperandTileInfo;

std::ostream &operator<<(std::ostream &os, const LoopTileInfo &lti);
std::ostream &operator<<(std::ostream &os, const OperandTileInfo &oti);

// get estimated cycle count for this kernel, given tiling config
mlir::LogicalResult getCost(mlir::Operation *rootOp,
                            llvm::SmallVector<int64_t> &tileSizes,
                            llvm::SmallVector<int64_t> &interchange,
                            llvm::SmallVector<int64_t> &out, std::string &errs);

bool gatherInfoLoopByLoop(mlir::linalg::LinalgOp &op,
                    llvm::ArrayRef<int64_t> &tileSizes,
                    llvm::ArrayRef<int64_t> &interchange,
                    llvm::SmallVector<OperandTileInfo> &out, std::string &errs);

// helper function to convert a small vector id int64_t to a string
void printSmallVector(llvm::SmallVector<int64_t> v, std::stringstream &ss);

/*
Name: getOperandDimPairsToTileInOrder
Description: Given an interchange array, populate a list of the relevant
operands to be tiled in each dimension.

    For example, given

        - a linalg operation with 3 Operands as follows:
            Operand 0: Shape 1x161, affine map (d0, d1, d2) -> (d0, d2)    //
let's nickname it I Operand 1: Shape 400x161, affine map (d0, d1, d2) -> (d1,
d2)  // let's nickname it W Operand 2: Shape 1x400, affine map (d0, d1, d2) ->
(d0, d1)    // let's nickname it O

        - an Interchange Array [0,1,2]

    This function returns the list [[(0,0),(1,0)], [(2,0),(2,1)], [(0,1),(2,1)]]
    with the format
    [[(operand #, position of dim 0 in operand)*],
    [(operand #, position of dim 1 in operand)*],
    [(operand #, position of dim 2 in operand)*]].

    Or with nicknames, the list is [[(I,0),(O,0)], [(W,0),(O,1)],
[(I,1),(W,1)]].

*/
// bool getOperandDimPairsToTileInOrder(
//     const mlir::linalg::LinalgOp &op,
//     const llvm::SmallVector<int64_t> &interchange,
//     llvm::SmallVector<
//         llvm::SmallVector<std::pair<mlir::OpOperand *, int64_t>>>);

}; // namespace myrtle