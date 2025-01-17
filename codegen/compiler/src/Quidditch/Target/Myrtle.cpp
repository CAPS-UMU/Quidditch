/*
Simple Analytical Cost Model for Tiling in Quidditch
*/

#include "Myrtle.h"

using namespace mlir;

namespace myrtle {

mlir::LogicalResult getCost(mlir::Operation *rootOp,
                            llvm::SmallVector<int64_t> &tileSizes,
                            llvm::SmallVector<int64_t> &interchange,
                            llvm::SmallVector<int64_t> &out,
                            std::string &errs) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      // only handle matmul transpose operations for now
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        llvm::ArrayRef<int64_t> tiles = llvm::ArrayRef<int64_t>(tileSizes);
        const auto &dims = op.createFlatListOfOperandStaticDims();

        if (dims.size() != 6 || (dims[0] != 1)) {
          std::stringstream ss;
          ss << "\nMyrtle: Only supporting 1D vector - 2D matrix transpose "
                "operations "
                "at "
                "the moment.\n";
          errs = ss.str();
          return failure();
        }

        // debugging
        std::stringstream ss;
        ss << "Dims:[";
        for (const auto &dim : dims) {
          ss << " " << dim;
          out.push_back(dim);
        }
        ss << " ]\n";

        // print per operand!
        ss << "\n Input Operands...\n";
        const auto &inputs = op.getRegionInputArgs();
        for (const auto &arg : inputs) {
          mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);
          const auto &shape = op.getShape(operand);
          ss << "\noperand # " << operand->getOperandNumber()
             << " with shape:[ ";
          for (const auto &num : shape) {
            ss << num << " ";
          }
          ss << " ]\n";
          // now print its map!!
          const auto &map = op.getMatchingIndexingMap(operand);
          std::string os = "";
          llvm::raw_string_ostream ros = llvm::raw_string_ostream(os);
          map.print(ros);
          ros << "\n";
          ros << "... which applied to tile sizes is ";
          const auto &relevantSizes = applyPermutationMap(map, tiles);
          ros << "[ ";
          for (const auto &sz : relevantSizes) {
            ros << sz << " ";
          }
          ros << " ]\n";
          ss << ros.str();

          // now apply tile sizes to the map??? Is it possible??
          // template <typename T> SmallVector<T> applyPermutationMap(AffineMap
          // map, llvm::ArrayRef<T> source)
          //    ArrayRef (const SmallVectorTemplateCommon< U *, DummyT > &Vec,
          //    std::enable_if_t< std::is_convertible< U *const *, T const *
          //    >::value > *=nullptr)
          // Construct an ArrayRef<const T*> from a SmallVector<T*>.
        }

        ss << "\n Output Operands...\n";
        const auto &outputs = op.getRegionOutputArgs();
        for (const auto &arg : outputs) {
          mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);
          const auto &shape = op.getShape(operand);
          ss << "\noperand # " << operand->getOperandNumber()
             << " with shape:[ ";
          for (const auto &num : shape) {
            ss << num << " ";
          }
          ss << " ]\n";
          // now print its map!!
          const auto &map = op.getMatchingIndexingMap(operand);
          std::string os = "";
          llvm::raw_string_ostream ros = llvm::raw_string_ostream(os);
          map.print(ros);
          ros << "\n";
          ros << "... which applied to tile sizes is ";
          const auto &relevantSizes = applyPermutationMap(map, tiles);
          ros << "[ ";
          for (const auto &sz : relevantSizes) {
            ros << sz << " ";
          }
          ros << " ]\n";
          ss << ros.str();
        }

        ss << "\nIn what order do we tile the dimensions?\n";
        for (const auto &order : interchange) {
          //    ss << "we tile with size " << tileSizes[order] << "\n";
          ss << "we tile dimension # " << order << "...\n";
          llvm::SmallVector<std::pair<Value, unsigned>> operandDimPairs =
              llvm::SmallVector<std::pair<Value, unsigned>>(0);
          op.mapIterationSpaceDimToAllOperandDims(order, operandDimPairs);
          for (const auto &pear : operandDimPairs) {
            // const auto &shape = op.getShape(pear.first);
            Value firstOperand = pear.first;
            unsigned firstOperandDim = pear.second;
            // Trivial case: `dim` size is available in the operand type.
            int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                                  .getShape()[firstOperandDim];
            if (ShapedType::isDynamic(dimSize)) {
              ss << "\ndimension of operand is dynamic. we cannot handle this "
                    "right now\n";
              errs = ss.str();
              return failure();
            }
            ss << "which means we tile operand # _'s" << firstOperandDim
               << "'s dimension with cardinality " << dimSize << "\n";

            const auto &inputs = op.getRegionInputArgs();
            for (const auto &arg : inputs) {
              mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);
            
              if(operand->is(firstOperand)){
                ss << "\nwhich is to say operand # " << operand->getOperandNumber() << "\n";
              }
            }
          }
          // void mapIterationSpaceDimToAllOperandDims(unsigned dimPos,
          // mlir::SmallVectorImpl<std::pair<Value, unsigned>>&
          // operandDimPairs); find all operands defined on this dimension

          //   void mapIterationSpaceDimToAllOperandDims(unsigned dimPos,
          // mlir::SmallVectorImpl<std::pair<Value, unsigned>>&
          // operandDimPairs); which means we tile operand __'s dimension __:
        }
     

        errs = ss.str();
        return failure();

        // return success();
      })
      .Default([&](const auto &op) {
        std::stringstream ss;
        ss << "\nMyrtle: Only supporting matmul transpose operations at "
              "the moment.\n";
        errs = ss.str();
        return failure();
      });
}

bool getOperandDimPairsToTileInOrder(
    const mlir::linalg::LinalgOp &op,
    const llvm::SmallVector<int64_t> &interchange,
    llvm::SmallVector<
        llvm::SmallVector<std::pair<mlir::OpOperand *, int64_t>>>) {
  return false;
}

void printSmallVector(llvm::SmallVector<int64_t> v, std::stringstream &ss) {
  ss << "[ ";
  for (int64_t e : v) {
    ss << e << " ";
  }
  ss << "]";
}






}; // namespace myrtle







/* Notes

   // /// Return the input block arguments of the region.
        // Block::BlockArgListType getRegionInputArgs();

        // /// Return the output block arguments of the region.
        // Block::BlockArgListType getRegionOutputArgs();

        // ArrayRef<int64_t> getShape(OpOperand * opOperand);

        // /// Return the block argument for an `opOperand`.
        // BlockArgument getMatchingBlockArgument(OpOperand * opOperand);

        // /// Return the operand for a `blockArgument`.
        // OpOperand *getMatchingOpOperand(BlockArgument blockArgument);

        // /// Return the input or output indexing map for `opOperand`.
        // AffineMap getMatchingIndexingMap(OpOperand * opOperand);

        // /// Return the indexing map for a `result`.
        // AffineMap getIndexingMapMatchingResult(OpResult result);
        // /// Return the value yielded by the region corresponding to an output
        // /// `opOperand`.

        /// Given a dimension of the iteration space of a Linalg operation,
        /// finds an
        /// operand in the operation that is defined on such dimension. Returns
        /// whether such operand was found or not. If found, also returns the
        /// operand value and the dimension position within the operand.
        // LogicalResult mapIterationSpaceDimToOperandDim(unsigned dimPos,
        // ::mlir::Value & operand, unsigned & operandDimPos);

        /// Given a dimension of the iteration space of a Linalg operation,
        /// finds all the operands in the operation that are defined on such
        /// dimension. Returns all the operand values found and their dimension
        /// positions in `operandDimPairs`.
        // void mapIterationSpaceDimToAllOperandDims(unsigned dimPos,
        // mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs);

        //         raw_string_ostream (std::string &O)

        // hoodle vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        // ss << "\n getIndexingMapsArray...\n";
        // std::string os = "";
        // llvm::raw_string_ostream ros = llvm::raw_string_ostream(os);

        // SmallVector<AffineMap> maps = op.getIndexingMapsArray();

        // for (const auto &m : maps) {
        //   // rootOp.emitWarning()
        //   m.print(ros);
        //   ros << "\n";
        // }
        // ss << ros.str();
        // hoodle ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  /// Applies composition by the dims of `this` to the integer `values` and
  /// returns the resulting values. `this` must be symbol-less.
  SmallVector<int64_t, 4> compose(ArrayRef<int64_t> values) const;

/home/hoppip/Quidditch/iree/third_party/torch-mlir/externals/llvm-project/mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp

  /// Return the operand for a `blockArgument`.
  // OpOperand *getMatchingOpOperand(BlockArgument blockArgument);
  const auto& outputs = op.getRegionOutputArgs();
  for(const auto& arg : outputs){
    const auto& operand = op.getMatchingOpOperand(arg);
    const auto& shape = op.getShape(operand);
    ss << "\nan output shape:[ ";
    for(const auto& num : shape){
        ss << num << " ";

    }
    ss << " ]\n";
  }

  /// Return the output block arguments of the region.
  // Block::BlockArgListType getRegionOutputArgs();

  /// Return the `opOperand` shape or an empty vector for scalars or vectors
  /// not wrapped within a tensor or a memref.
  // ArrayRef<int64_t> getShape(OpOperand* opOperand);

           ::mlir::Operation::operand_range getInputs() {
    return getODSOperands(0);
  }

  ::mlir::Operation::operand_range getOutputs() {
    return getODSOperands(1);
  }
  linalg op reference:
  /home/hoppip/Quidditch/build3.10/codegen/iree-configuration/iree/llvm-project/tools/mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h.inc


  // very manual but we will code it this way first
        // SmallVector<int64_t> input(2, 0);
        // SmallVector<int64_t> weight(2, 0);
        // SmallVector<int64_t> output(2, 0);
        // input[0] = dims[0];
        // input[1] = dims[1];
        // weight[0] = dims[2];
        // weight[1] = dims[3];
        // output[0] = dims[4];
        // output[1] = dims[5];

        // a dimension

        // b dimension

        // c dimension
 // SmallVector<int64_t> inputBnds(2, 0);
        // SmallVector<int64_t> weightBnds(2, 0);
        // SmallVector<int64_t> outputBnds(2, 0);
        // inputBnds[0] = 1 ;//dims[0];
        // inputBnds[1] = dims[1]/tileSizes[1];
        // weightBnds[0] = dims[2]/tileSizes[1];
        // weightBnds[1] = dims[3]/tileSizes[2];
        // outputBnds[0] = 1;//dims[4];
        // outputBnds[1] = dims[5]/tileSizes[2];

*/