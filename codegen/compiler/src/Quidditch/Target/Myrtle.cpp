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
        std::stringstream ss;
        llvm::ArrayRef<int64_t> tiles = llvm::ArrayRef<int64_t>(tileSizes);
        llvm::ArrayRef<int64_t> loopInterchange =
            llvm::ArrayRef<int64_t>(interchange);

        llvm::SmallVector<OperandTileInfo> operands = {};
        std::string errStr = "";
        std::string gatherInfoStr = "";
        if (!gatherInfoLoopByLoop(op, tiles, loopInterchange, operands, gatherInfoStr)) {
          errs = gatherInfoStr;
          return failure();
        }
        for(auto& oper : operands){
            oper.computeValues(errStr);
            ss << errStr;
        }
        out.push_back(operands[1].cycles);

        // extremely snitch-specific cost model calculations now
        // get cycle count from second argument
        // get # of L1 to RF transfers per core???

        // errs = ss.str();
        //  errs = errStr;
        errs = gatherInfoStr;
        // return failure();

        return success();
      })
      .Default([&](const auto &op) {
        std::stringstream ss;
        ss << "\nMyrtle: Only supporting matmul transpose operations at "
              "the moment.\n";
        errs = ss.str();
        return failure();
      });
}

void printSmallVector(llvm::SmallVector<int64_t> v, std::stringstream &ss) {
  ss << "[ ";
  for (int64_t e : v) {
    ss << e << " ";
  }
  ss << "]";
}

void myrtle::OperandTileInfo::tileLoopByLoop(
    mlir::linalg::LinalgOp &op, llvm::ArrayRef<int64_t> &tileSizes,
    llvm::ArrayRef<int64_t> &interchange, MEMSPACE innermostLoopMemspace) {
  const auto &map = op.getMatchingIndexingMap(operand);
  const auto &originalShape = op.getShape(operand);
  for (const auto &order : interchange) {
    // Set default tile size and tile count.
    // If this is the first dimension to be tiled,
    // set tile shape to original operand size.
    // Otherwise, use most recent tile size.
    myrtle::LoopTileInfo info(
        order, 1, L3,
        (loops.empty() ? originalShape : loops[loops.size() - 1].tileShape));
    if (tileSizes[order] != 0) { // check that the tile size is valid
      // check that this operand can indeed be tiled in this dimension
      const auto &res =
          map.getResultPosition(getAffineDimExpr(order, map.getContext()));
      if ((res != std::nullopt)) {
        // TODO: handle padding!!!
        info.tileCount = info.tileShape[*res] / tileSizes[order];
        info.tileShape[*res] = tileSizes[order];
      }
    }
    loops.push_back(info);
  }
  loops[loops.size() - 1].mem = innermostLoopMemspace;
}

void myrtle::OperandTileInfo::tileOneLoopMore(mlir::linalg::LinalgOp &op,
                                              int64_t loopBound, int64_t dim,
                                              MEMSPACE memspace) {
  const auto &map = op.getMatchingIndexingMap(operand);
  const auto &originalShape = op.getShape(operand);
  // If this is the first dimension to be tiled,
  // set tile shape to original operand size.
  // Otherwise, use most recent tile size.
  myrtle::LoopTileInfo info(
      dim, 1, memspace,
      (loops.empty() ? originalShape : loops[loops.size() - 1].tileShape));
  // if bound is not valid, return
  if (loopBound <= 0) {
    return;
  }
  // check that this operand can indeed be tiled in this dimension
  const auto &res =
      map.getResultPosition(getAffineDimExpr(dim, map.getContext()));
  if ((res != std::nullopt)) {
    // TODO: handle padding!!!
    info.tileCount = loopBound;
    info.tileShape[*res] = info.tileShape[*res] / loopBound;
  }

  loops.push_back(info);
}


void myrtle::OperandTileInfo::computeValues(std::string& errs){
    // compute cycles
    int64_t cost_of_fmadd = 3;
    size_t row_dim_index = 0;
    size_t reduction_dim_index = 1;
    std::stringstream ss;
    int64_t tileCountProd = 1;
    for(const auto& info : loops){
        tileCountProd *= info.tileCount;
    }
    const auto& innerMost = loops[loops.size()-1];
    ss << "Compute Cycles:\n";
    ss << "\tRow dim: " << innerMost.tileShape[row_dim_index] << " elts\n";
    ss << "\tCol (reduction) dim: " << innerMost.tileShape[reduction_dim_index] << " elts\n";
    int64_t fmaddCountPerCore = innerMost.tileShape[row_dim_index] * innerMost.tileShape[reduction_dim_index];
    int64_t cycleCountPerCore = fmaddCountPerCore * cost_of_fmadd;
    cycles = tileCountProd * cycleCountPerCore;
    ss << "\tfmaddCountPerCore = " << fmaddCountPerCore << " \n";
    ss << "\tcycleCountPerCore = " << cycleCountPerCore << " \n";
    ss << "\tcycles = "<< tileCountProd  << " * " << cycleCountPerCore << " = "<<cycles << "\n\n";
    errs = ss.str();
}

// assumes the linalg operation is a matmul_transpose_b
// "tiles" the operands one dimension at a time,
// saving tile sizes and tile counts for each dimension
// as well as the Memory Space where each tile should be located
bool gatherInfoLoopByLoop(mlir::linalg::LinalgOp &op,
                    llvm::ArrayRef<int64_t> &tileSizes,
                    llvm::ArrayRef<int64_t> &interchange,
                    llvm::SmallVector<OperandTileInfo> &out,
                    std::string &errs) {
  std::stringstream ss; // for error logging
  // TODO: get rid of hardcoding!!!
  // could make it a little more generic by saying "the non-reduction dimension
  // of the second arg of matmultranspose"
  int64_t rowDimension = 1;
  int64_t numCores = 8;

  const auto &inputs = op.getRegionInputArgs();
  //  for each input operand
  for (const auto &arg : inputs) {
    OperandTileInfo oper = OperandTileInfo(op.getMatchingOpOperand(arg));
    // tile to fit in L1
    oper.tileLoopByLoop(op, tileSizes, interchange, L1);
    // tile for compute cores
    oper.tileOneLoopMore(op, numCores, rowDimension, L1);
    out.push_back(oper);
    ss << "\n" << oper;
  }
  const auto &outputs = op.getRegionOutputArgs();
  // for each output operand
  for (const auto &arg : outputs) {
    OperandTileInfo oper = OperandTileInfo(op.getMatchingOpOperand(arg));
    // tile to fit in L1
    oper.tileLoopByLoop(op, tileSizes, interchange, L1);
    // tile for compute cores
    oper.tileOneLoopMore(op, numCores, rowDimension, L1);
    out.push_back(oper);
    ss << "\n" << oper;
  }
  errs = ss.str();
  return true;
}

std::ostream &operator<<(std::ostream &os, const LoopTileInfo &lti) {
  os << "LoopTileInfo:\n";
  os << "\t dim: " << lti.dim << "\n";
  os << "\t tile shape: [ ";
  for (const auto &sz : lti.tileShape) {
    os << sz << " ";
  }
  os << "]\n";
  os << "\t tile count: " << lti.tileCount << "\n";
  os << "\t mem: " << ((lti.mem == L1) ? "L1" : "L3") << "\n";
  return os;
}
std::ostream &operator<<(std::ostream &os, const OperandTileInfo &oti) {
  os << "OperandTileInfo:\n";
  os << "\tLoops:[ ";
  for (const auto &loop : oti.loops) {
    os << "\n\t " << loop;
  }
  os << "\t]\n";
  return os;
}

}; // namespace myrtle

/* Notes


 SmallVector<unsigned> parallelDims = {};
 SmallVector<unsigned> reductionDims = {};
 op.getParallelDims(parallelDims);
 op.getReductionDims(reductionDims);
ss << "number of parallel loops is " << op.getNumParallelLoops() << "\n";
for(const auto& dim : parallelDims){
    ss << "parallel dim " << dim << "\n";

}
ss << "number of reduction loops is " << op.getNumReductionLoops() << "\n";
for(const auto& dim : reductionDims){
    ss << "reduction dim " << dim << "\n";

}

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

// freezing a version of the "get cost" function below:
/*

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

        ss << "\nIn what order do we tile the dimensions? *********** v\n";
        // AffineMap mlir::linalg::LinalgOp::getShapesToLoopsMap()
        std::string os = "";
        llvm::raw_string_ostream ros = llvm::raw_string_ostream(os);
        ros << "shapesToLoopsMap: ";
        op.getShapesToLoopsMap().print(ros);
        ros << "\nloopsToShapeMap: ";
        op.getLoopsToShapesMap().print(ros);
        ros << "\n... which applied to tile sizes is ";
        const auto &relevantSizes =
            applyPermutationMap(op.getLoopsToShapesMap(), tiles);
        ros << "[ ";
        for (const auto &sz : relevantSizes) {
          ros << sz << " ";
        }
        ros << " ]\n";

        ss << ros.str() << "\n hoodle \n";

        // bool isFunctionOfDim
        /// Extracts the first result position where `input` dimension resides.
        /// Returns `std::nullopt` if `input` is not a dimension expression or
        /// cannot be found in results.
        // std::optional<unsigned> getResultPosition(AffineExpr input) const;
        //  std::optional<unsigned> getResultPosition(AffineExpr input) const;
        // getAffineDimExpr(dimPos, idxMap.getContext())))

        for (const auto &order : interchange) {
          for (const auto &arg : inputs) {
            mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);
            const auto &map = op.getMatchingIndexingMap(operand);
            // if (map.isFunctionOfDim(order)) {
            //   ss << "operand # " << operand->getOperandNumber()
            //      << " DOES contain dimension " << order << " in its
results.\n";
            // }
            const auto & res = map.getResultPosition(getAffineDimExpr(order,
map.getContext())); if(res!= std::nullopt){ ss << "operand # " <<
operand->getOperandNumber()
                 << " DOES contain dimension " << order << " in its results:
position "<< *res<<"\n";
            }
          }
          for (const auto &arg : outputs) {
            mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);
            const auto &map = op.getMatchingIndexingMap(operand);
             const auto & res = map.getResultPosition(getAffineDimExpr(order,
map.getContext())); if(res!= std::nullopt){ ss << "operand # " <<
operand->getOperandNumber()
                 << " DOES contain dimension " << order << " in its results:
position "<< *res<<"\n";
            }
            // if (map.isFunctionOfDim(order)) {
            //   ss << "operand # " << operand->getOperandNumber()
            //      << " DOES contain dimension " << order << " in its
results."<<"\n";
            // }
          }
        }
        // mymap.insert ( std::pair<char,int>('a',100) );
        //  std::map<mlir::Value*,myrtle::OperandTileInfo> val2OpMap = {};
        //  for(const auto& arg : inputs){
        //      mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);
        //
val2OpMap.insert(std::pair<mlir::Value*,myrtle::OperandTileInfo>(&operand->get(),OperandTileInfo(operand)));
        //  }
        //  for(const auto& output : outputs){
        //      mlir::OpOperand *const operand =
        //      op.getMatchingOpOperand(output);
        //
val2OpMap.insert(std::pair<mlir::Value*,myrtle::OperandTileInfo>(&operand->get(),OperandTileInfo(operand)));
        //  }
        ss << "\nIn what order do we tile the dimensions? *********** ^\n";

        // not going to touch what works
        ss << "\nIn what order do we tile the dimensions?\n";
        for (const auto &order : interchange) {
          //    ss << "we tile with size " << tileSizes[order] << "\n";
          // AffineMap mlir::linalg::LinalgOp::getLoopsToShapesMap()
          ss << "we tile dimension # " << order << "...\n";
          llvm::SmallVector<std::pair<Value, unsigned>> operandDimPairs =
              llvm::SmallVector<std::pair<Value, unsigned>>(0);
          op.mapIterationSpaceDimToAllOperandDims(order, operandDimPairs);
          for (const auto &pear : operandDimPairs) {
            // const auto &shape = op.getShape(pear.first);
            Value firstOperand = pear.first;
            unsigned firstOperandDim = pear.second;
            // Trivial case: `dim` size is available in the operand type.
            // int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
            //                       .getShape()[firstOperandDim];
            // if (ShapedType::isDynamic(dimSize)) {
            //   ss << "\ndimension of operand is dynamic. we cannot handle this
            //   "
            //         "right now\n";
            //   errs = ss.str();
            //   return failure();
            // }
            // ss << "which means we tile operand # _'s" << firstOperandDim
            //    << "'s dimension with cardinality " << dimSize << "\n";

            const auto &inputs = op.getRegionInputArgs();
            for (const auto &arg : inputs) {
              mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);

              if (operand->is(firstOperand)) {
                int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                                      .getShape()[firstOperandDim];
                if (ShapedType::isDynamic(dimSize)) {
                  ss << "\ndimension of operand is dynamic. we cannot handle "
                        "this "
                        "right now\n";
                  errs = ss.str();
                  return failure();
                }

                ss << "which is to say tile operand # "
                   << operand->getOperandNumber()
                   << "'s dimension with cardinality " << dimSize
                   << " by tile size " << tileSizes[order] << "\n";
              }
            }
            const auto &outputs = op.getRegionOutputArgs();
            for (const auto &arg : outputs) {
              mlir::OpOperand *const operand = op.getMatchingOpOperand(arg);

              if (operand->is(firstOperand)) {
                int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                                      .getShape()[firstOperandDim];
                if (ShapedType::isDynamic(dimSize)) {
                  ss << "\ndimension of operand is dynamic. we cannot handle "
                        "this "
                        "right now\n";
                  errs = ss.str();
                  return failure();
                }

                ss << "which is to say tile operand # "
                   << operand->getOperandNumber()
                   << "'s dimension with cardinality " << dimSize
                   << " by tile size " << tileSizes[order] << "\n";
              }
            }
          }
          // void mapIterationSpaceDimToAllOperandDims(unsigned dimPos,
          // mlir::SmallVectorImpl<std::pair<Value, unsigned>>&
          // operandDimPairs); find all operands defined on this dimension

          //   void mapIterationSpaceDimToAllOperandDims(unsigned dimPos,
          // mlir::SmallVectorImpl<std::pair<Value, unsigned>>&
          // operandDimPairs); which means we tile operand __'s dimension __:
          // tileSizeIndex++;
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

*/