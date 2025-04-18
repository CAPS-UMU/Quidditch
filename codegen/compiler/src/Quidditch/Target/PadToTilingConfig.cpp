#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_PADTOTILINGCONFIGPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace quidditch::Snitch;
using namespace mlir::iree_compiler;

namespace {
class PadToTilingConfig
    : public quidditch::impl::PadToTilingConfigPassBase<PadToTilingConfig> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

/// Returns true if it is legal to zero-pad the given linalg operation.
/// Legal is defined as being able to extend the iteration space and the
/// corresponding operands using zero-padding without changing the values
/// in the slice corresponding to the ops original result.
static bool canZeroPad(linalg::LinalgOp linalgOp) {
  // Elementwise operations can be padded with any value as there are no cross
  // dimension data dependencies.
  if (linalgOp.getNumParallelLoops() == linalgOp.getNumLoops())
    return true;

  // Contractions can be zero padded.
  if (linalg::isaContractionOpInterface(linalgOp))
    return true;

  // Convolutions can be zero padded.
  return linalg::isaConvolutionOpInterface(linalgOp);
}

static LogicalResult padToMultipleOf(linalg::LinalgOp &linalgOp,
                                     SmallVector<int64_t> config) {
  for (int64_t &value : config)
    if (value == 0)
      value = 1;

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(
              llvm::to_vector(llvm::seq<int64_t>(config.size())))
          .setPadToMultipleOf(config)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

  auto loweringConfig = getLoweringConfig<LoweringConfigAttr>(linalgOp);

  auto builder = IRRewriter(linalgOp);
  SmallVector<tensor::PadOp> padOps;
  linalg::LinalgOp oldOp = linalgOp;
  SmallVector<Value> replacements;
  if (failed(linalg::rewriteAsPaddedOp(builder, linalgOp, options, linalgOp,
                                       replacements, padOps)))
    return failure();
  builder.replaceOp(oldOp, replacements);

  if (loweringConfig)
    setLoweringConfig(linalgOp, loweringConfig);
  return success();
}

static LogicalResult padToTileSize(linalg::LinalgOp &linalgOp,
                                   std::optional<IntegerAttr> computeCores) {
  SmallVector<int64_t> tileSize;
  if (auto loweringConfig = getLoweringConfig<LoweringConfigAttr>(linalgOp)) {
    for (auto getTileSizeMem : {&LoweringConfigAttr::getWorkgroupTiles,
                                &LoweringConfigAttr::getL1Tiles}) {
      tileSize = llvm::to_vector((loweringConfig.*getTileSizeMem)());
      size_t numLoops = linalgOp.getNumLoops();
      while (tileSize.size() < numLoops)
        tileSize.push_back(0);

      if (failed(padToMultipleOf(linalgOp, tileSize)))
        return failure();
    }
  }

  if (!computeCores)
    return success();

  // TODO: This duplicates the logic for thread tiling risking them to go out of
  //       sync.
  //       We probably want 'LoweringConfigAttr' to include these tile sizes
  //       in the future as well.
  if (tileSize.empty())
    tileSize = linalgOp.getStaticLoopRanges();

  std::optional<unsigned> largestParallelDim;
  std::optional<int64_t> largestParallelSize;
  for (auto [index, iterType, range] :
       llvm::enumerate(linalgOp.getIteratorTypesArray(), tileSize)) {
    // Not doing reduction tiling.
    if (iterType == utils::IteratorType::reduction) {
      range = 0;
      continue;
    }

    // Not tileable.
    if (range <= 1) {
      range = 0;
      continue;
    }

    // Not tiling dynamic dimensions right now.
    if (range == ShapedType::kDynamic) {
      range = 0;
      continue;
    }

    if (!largestParallelSize || range > largestParallelSize) {
      largestParallelDim = index;
      largestParallelSize = range;
    }
  }

  if (largestParallelDim) {
    assert(largestParallelSize);
    tileSize[*largestParallelDim] = llvm::divideCeil(
        *largestParallelSize, computeCores->getValue().getSExtValue());
  }

  return padToMultipleOf(linalgOp, std::move(tileSize));
}

/// Returns true if the given pad operation uses an undefined value as padding
/// value.
static bool hasUndefPadding(tensor::PadOp padOp) {
  Value constant = padOp.getConstantPaddingValue();
  return constant &&
         matchPattern(constant, m_Constant<ub::PoisonAttrInterface>(nullptr));
}

/// Clones 'padOp' using 'rewriter' and replaces its padding value with an
/// undefined value.
static tensor::PadOp cloneWithUndefPad(PatternRewriter &rewriter,
                                       tensor::PadOp padOp) {
  auto newPadOp = cast<tensor::PadOp>(rewriter.cloneWithoutRegions(*padOp));
  {
    OpBuilder::InsertionGuard guard{rewriter};
    rewriter.setInsertionPointToEnd(&newPadOp.getRegion().emplaceBlock());
    for (unsigned _ : llvm::seq(padOp.getSource().getType().getRank()))
      newPadOp.getRegion().addArgument(rewriter.getIndexType(),
                                       padOp->getLoc());

    // TODO: This is very wrong as poison is stronger than undef as there are
    //       operations where a poison value will cause immediate undefined
    //       behaviour where an undef value wouldn't.
    //       Our lowering does the equivalent of using an undef value for now
    //       but things like folding won't respect it.
    //       The correct fix would be to have `ub.freeze` upstream.
    Value poison = rewriter.create<ub::PoisonOp>(
        newPadOp->getLoc(), padOp.getType().getElementType());
    rewriter.create<tensor::YieldOp>(padOp->getLoc(), poison);
  }
  return newPadOp;
}

namespace {

// helper functions for pattern to remove redundant buffer

bool checkSliceIsRedundant(tensor::ExtractSliceOp &sliceOp,
                           std::stringstream &ss);

bool checkFillsWithZeros(linalg::FillOp &fillOp, std::stringstream &ss);

bool checkPadsWithZeros(mlir::tensor::PadOp &padOp, std::stringstream &ss);

bool checkIsMatVecTrans(linalg::LinalgOp &linalgOp);

bool checkOutputIsPadded(linalg::LinalgOp &linalgOp);

bool isPaddingFillWithZeros(mlir::tensor::PadOp &padOp, std::stringstream &ss);

bool isPaddingZeroOffsetSlice(mlir::tensor::PadOp &padOp,
                              std::stringstream &ss);

bool slicesFillWithZeros(tensor::ExtractSliceOp &extractSlice,
                         std::stringstream &ss);

bool fillsEmptyTensor(mlir::linalg::FillOp &fill, std::stringstream &ss);

bool replaceEmptyWithLargerEmpty(PatternRewriter &rewriter,
                                 linalg::LinalgOp &linalgOp,
                                 tensor::EmptyOp *largerEmpty,
                                 std::stringstream &ss);

bool replaceLinalgOutputWithLargerFill(PatternRewriter &rewriter,
                                       linalg::LinalgOp &linalgOp,
                                       mlir::linalg::FillOp *largerFill,
                                       std::stringstream &ss);

bool replaceFillWithLargerFill(PatternRewriter &rewriter,
                               linalg::LinalgOp &linalgOp,
                               tensor::EmptyOp *&largerEmpty,
                               std::stringstream &ss);

bool replacePadOpWithLargerFill(PatternRewriter &rewriter,
                                linalg::LinalgOp &linalgOp,
                                std::stringstream &ss);
/*
plan for more general pattern:
1. maybe check if I am a matrix-vector-transpose
what is the size of my output argument, o_sz?
what is the defining OP of my output argument, o_arg?
func(o_arg,o_sz, filledWZeroAtSz=false)
case: fill
  A. fill with 0, size of o_sz?
    Recurse(definingOp, o_sz, true)
  B. No.
case: slice
  A. redundant slice?
    Recurse(definingOp, o_sz)
  B. a slice into something bigger? No.
case: pad
  A: pad with 0, to size of o_sz?
    pad_sz = size of pad
    Recurse(definingOp, o_sz)
  B: No.
case: empty
  A. sz <= o_sz
    if(filledWZeroAtSz) then REPLACE
    else: No.
  B No.
*/

LogicalResult reducesToEmpty(const mlir::Operation &op) { return failure(); }

LogicalResult reducesToPaddedEmpty(mlir::Operation &op) { 
  TypeSwitch<Operation *, LogicalResult>(&op)//fill.getOperand(theIndex)
  .Case<mlir::tensor::PadOp>(
      [&](mlir::tensor::PadOp op) { 
        return reducesToEmpty(*op.getOperand(0).getDefiningOp()); })
  .Case<linalg::FillOp>(
        [&](linalg::FillOp op) { return failure(); })
  .Case<tensor::EmptyOp>(
          [&](tensor::EmptyOp op) { return failure(); })
  .Case<tensor::ExtractSliceOp>(
            [&](tensor::ExtractSliceOp op) { return failure(); })
  .Default([&](const auto &op) { return failure(); }); //tensor::ExtractSliceOp
 }

// this pattern runs BEFORE all of the undefpadding patterns.
// It removes redundant buffer "allocated" during padding of the row dimension.
/*
  If I have a matrix-vector transpose operation,
  and my OUTPUT ARGUMENT is a tensor.pad operation, where
             - the padding is set to zero
             - the tensor to padd is a slice called SLICE, where
             - SLICE is a _redundant_ slice of a fill operation called FILL,
  where
             - FILL fills a tensor.empty operation with _zeros_
  Then
   1. create tensor.empty of size _padded_
   2. fill empty tensor with zeros
   3. Replace linalg op's OUTPUT ARGUMENT with this tensor.

*/

struct RemoveRedundantBuffer : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {

    // changed initialized to false
    bool changed = false;

    // only handle matmul transpose operations for now
    if (!checkIsMatVecTrans(linalgOp)) {
      return success(changed);
    }

    // don't handle linalg operations with multiple output arguments
    const auto &outputs = linalgOp.getRegionOutputArgs();
    if (outputs.vec().size() != 1) {
      linalgOp.emitWarning()
          << "Error: Expected a mat-vec-transb with only ONE output argument\n";
      return success(changed);
    }

    // make sure the output argument is a pad.tensor
    if (!checkOutputIsPadded(linalgOp)) {
      return success(changed);
    }

    // extract the output argument
    mlir::BlockArgument &outputArg = linalgOp.getRegionOutputArgs()[0];
    mlir::OpOperand *oper = linalgOp.getMatchingOpOperand(outputArg);
    mlir::Value operAsValue = oper->get();
    mlir::tensor::PadOp padOp = operAsValue.getDefiningOp<tensor::PadOp>();

    // TODO: if the output op derives from a padded empty tensor, replace with
    // larger tensor filled with 0
    
    // Right now, always replaces output arg with a larger tensor of zeroes
    // create empty op
    mlir::ValueRange no_operands({});
    tensor::EmptyOp empty = rewriter.create<tensor::EmptyOp>(
        padOp.getLoc(), padOp.getType(), no_operands);
    // create 0 op
    auto zeroVal = rewriter.getZeroAttr(getElementTypeOrSelf(padOp.getType()));
    arith::ConstantOp zConstant =
        rewriter.create<arith::ConstantOp>(empty.getLoc(), zeroVal);
    // create fill with 0 op
    llvm::SmallVector<mlir::Value> inputVals = {zConstant};
    llvm::SmallVector<mlir::Value> outputVals = {empty};
    linalg::FillOp fill = rewriter.create<linalg::FillOp>(
        zConstant.getLoc(), padOp.getType(), inputVals, outputVals);
    // replace output argument with the new fill operation
    rewriter.replaceAllOpUsesWith(padOp, fill);
    changed = true;
    return success(changed);
  }
};

bool checkSliceIsRedundant(tensor::ExtractSliceOp &sliceOp,
                           std::stringstream &ss) {
  return true;
}

bool checkFillsWithZeros(linalg::FillOp &fillOp, std::stringstream &ss) {
  return true;
}

bool checkPadsWithZeros(mlir::tensor::PadOp &padOp, std::stringstream &ss) {
  return true;
}

bool checkIsMatVecTrans(linalg::LinalgOp &linalgOp) {
  auto result = TypeSwitch<Operation *, LogicalResult>(linalgOp.getOperation())
                    .Case<linalg::MatmulTransposeBOp>(
                        [&](linalg::LinalgOp op) { return success(); })
                    .Default([&](const auto &op) { return failure(); });
  return (!failed(result));
}

bool checkOutputIsPadded(linalg::LinalgOp &linalgOp) {
  // make sure there is only ONE output argument
  const auto &outputs = linalgOp.getRegionOutputArgs();
  if (outputs.vec().size() != 1) {
    linalgOp.emitWarning()
        << "Error: Expected a mat-vec-transb with only ONE output argument\n";
    return false;
  }
  // make sure the output argument is a pad.tensor
  auto &outputArg = outputs[0];
  auto oper = linalgOp.getMatchingOpOperand(outputArg);
  auto padOp = oper->get().getDefiningOp<tensor::PadOp>();
  if (!padOp) {
    return false;
  }
  return true;
}

bool isPaddingFillWithZeros(mlir::tensor::PadOp &padOp, std::stringstream &ss) {
  auto fill = padOp.getSource().getDefiningOp<linalg::FillOp>();
  if (!fill) {
    // auto theSource = padOp.getSource();
    ss << "error: not padding a fill operation\n";
    return false;
  }
  return true;
}

bool isPaddingZeroOffsetSlice(mlir::tensor::PadOp &padOp,
                              std::stringstream &ss) {
  auto extractSlice = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!extractSlice) {
    return false;
  }
  return true;
}

bool slicesFillWithZeros(tensor::ExtractSliceOp &extractSlice,
                         std::stringstream &ss) {
  auto fill = extractSlice.getSource().getDefiningOp<linalg::FillOp>();
  if (!fill) {
    ss << "slice passed to pad op does not meet criteria (not slicing result "
          "of a fillOp)\n";
    return false;
  }
  // make sure this FILL was filling with ZEROs
  if (!checkFillsWithZeros(fill, ss)) {
    ss << "does not fill with zero\n";
    return false;
  }
  return true;
}

bool fillsEmptyTensor(mlir::linalg::FillOp &fill, std::stringstream &ss) {
  // make sure there is only ONE output argument
  const auto &fillOutputs = fill.getRegionOutputArgs();
  if (fillOutputs.vec().size() != 1) {
    ss << "Error: Expected a fill operation with only ONE output argument\n";
    return false;
  }
  // make sure the output argument is a tensor.empty
  const auto &fillOutputArg = fillOutputs[0];
  auto fillOper = fill.getMatchingOpOperand(fillOutputArg);
  auto emptyOp = fillOper->get().getDefiningOp<tensor::EmptyOp>();
  if (!emptyOp) {
    ss << "Error: fill operation does not fill an empty tensor.\n";
    return false;
  }
  return true;
}

bool replaceEmptyWithLargerEmpty(PatternRewriter &rewriter,
                                 linalg::LinalgOp &linalgOp,
                                 tensor::EmptyOp *largerEmpty,
                                 std::stringstream &ss) {
  mlir::BlockArgument &outputArg = linalgOp.getRegionOutputArgs()[0];
  mlir::OpOperand *oper = linalgOp.getMatchingOpOperand(outputArg);
  mlir::Value operAsValue = oper->get();
  mlir::tensor::PadOp padOp = operAsValue.getDefiningOp<tensor::PadOp>();
  auto extractSlice = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto fill = extractSlice.getSource().getDefiningOp<linalg::FillOp>();
  const auto &fillOutputs = fill.getRegionOutputArgs();
  const auto &fillOutputArg = fillOutputs[0];
  auto fillOper = fill.getMatchingOpOperand(fillOutputArg);
  auto emptyOp = fillOper->get().getDefiningOp<tensor::EmptyOp>();
  // copy output pad operation's type
  ReifiedRankedShapedTypeDims dims;
  if (failed(reifyResultShapes(rewriter, padOp, dims)))
    return false;

  rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
      emptyOp, dims.front(), getElementTypeOrSelf(padOp.getType()));

  return true;
}

bool replaceLinalgOutputWithLargerFill(PatternRewriter &rewriter,
                                       linalg::LinalgOp &linalgOp,
                                       mlir::linalg::FillOp *largerFill,
                                       std::stringstream &ss) {
  mlir::BlockArgument &outputArg = linalgOp.getRegionOutputArgs()[0];
  mlir::OpOperand *oper = linalgOp.getMatchingOpOperand(outputArg);
  mlir::Value operAsValue = oper->get();
  mlir::tensor::PadOp padOp = operAsValue.getDefiningOp<tensor::PadOp>();
  auto extractSlice = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto fill = extractSlice.getSource().getDefiningOp<linalg::FillOp>();
  auto fillAsOp = fill.getResult(0);
  std::string os = "";
  llvm::raw_string_ostream ros = llvm::raw_string_ostream(os);

  ros << "\ninspecting the getDpsInitsMutable collection \n";

  for (OpOperand &inits : linalgOp.getDpsInitsMutable()) {
    rewriter.modifyOpInPlace(linalgOp,
                             [&, &operand = inits] { operand.set(fillAsOp); });
  }

  return true;
}

bool replaceFillWithLargerFill(PatternRewriter &rewriter,
                               linalg::LinalgOp &linalgOp,
                               tensor::EmptyOp *&largerEmpty,
                               std::stringstream &ss) {
  mlir::BlockArgument &outputArg = linalgOp.getRegionOutputArgs()[0];
  mlir::OpOperand *oper = linalgOp.getMatchingOpOperand(outputArg);
  mlir::Value operAsValue = oper->get();
  mlir::tensor::PadOp padOp = operAsValue.getDefiningOp<tensor::PadOp>();
  auto extractSlice = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto fill = extractSlice.getSource().getDefiningOp<linalg::FillOp>();
  const auto &fillOutputs = fill.getRegionOutputArgs();
  const auto &fillOutputArg = fillOutputs[0];
  auto fillOper = fill.getMatchingOpOperand(fillOutputArg);
  auto emptyOp = fillOper->get().getDefiningOp<tensor::EmptyOp>();
  // create a singleton list of result types, where
  // result type is the same as the padOp's/ new emptyOp's
  llvm::SmallVector<mlir::Type> resultType = {};
  resultType.push_back(padOp.getType());
  llvm::SmallVector<mlir::Value> inputVals = {};
  const auto &fillInputs = fill.getRegionInputArgs();
  //  copy input operands as mlir values
  for (const auto &arg : fillInputs) {
    auto oper = fill.getMatchingOpOperand(arg);
    const auto theIndex = oper->getOperandNumber();
    const auto theVal = fill.getOperand(theIndex);
    inputVals.push_back(theVal);
  }
  //  copy output operands as mlir values
  llvm::SmallVector<mlir::Value> outputVals = {};
  outputVals.push_back(emptyOp);
  mlir::ValueRange inputValRange(inputVals);
  mlir::ValueRange outputValRange(outputVals);
  mlir::TypeRange resultTypeRange(resultType);

  rewriter.replaceOpWithNewOp<linalg::FillOp>(fill, resultTypeRange,
                                              inputValRange, outputValRange);
  return true;
}

bool replacePadOpWithLargerFill(PatternRewriter &rewriter,
                                linalg::LinalgOp &linalgOp,
                                std::stringstream &ss) {
  mlir::BlockArgument &outputArg = linalgOp.getRegionOutputArgs()[0];
  mlir::OpOperand *oper = linalgOp.getMatchingOpOperand(outputArg);
  mlir::Value operAsValue = oper->get();
  mlir::tensor::PadOp padOp = operAsValue.getDefiningOp<tensor::PadOp>();
  auto extractSlice = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto fill = extractSlice.getSource().getDefiningOp<linalg::FillOp>();
  rewriter.replaceAllOpUsesWith(padOp, fill);
  linalgOp.emitWarning() << "\nI did the rewrite!!";
  return true;
}

} // namespace

namespace {

// Patterns applied on linalg operations to turn zero pads created by the
// padding rewriter into undef pads.
// Note that these assume that padding of results are not consumed in a way
// where its semantics have any impact on the final output.
// In simplified words: The linalg's result is only used by the `extract_slice`
// which extracts the slice corresponding to the original unpadded computation.
//
// TODO: This might be cleaner to implement right after or during the padding
//       rewrite.

/// Optimizes every operand of an elementwise operation to be undef padded.
struct OptimizeElementwisePad : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Only elementwise operations.
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return failure();

    bool changed = false;
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      auto padOp = operand.get().getDefiningOp<tensor::PadOp>();
      if (!padOp)
        continue;

      if (hasUndefPadding(padOp))
        continue;

      auto newPadOp = cloneWithUndefPad(rewriter, padOp);
      rewriter.modifyOpInPlace(linalgOp, [&] { operand.set(newPadOp); });
      changed = true;
    }
    return success(changed);
  }
};

/// Optimizes the init operand of a contraction operation to be undef padded.
struct OptimizeContractionOutputPad
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(linalgOp))
      return failure();

    bool changed = false;
    for (OpOperand &operand : linalgOp.getDpsInitsMutable()) {
      auto padOp = operand.get().getDefiningOp<tensor::PadOp>();
      if (!padOp)
        continue;

      if (hasUndefPadding(padOp))
        continue;

      auto newPadOp = cloneWithUndefPad(rewriter, padOp);
      rewriter.modifyOpInPlace(linalgOp, [&] { operand.set(newPadOp); });
      changed = true;
    }
    return success(changed);
  }
};

/// Optimizes operands that are padded but only contribute to an output tensor
/// that is undef padded to also be undef padded.
struct PropagateUnusedOutputPads : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();

    bool changed = false;
    // For every operand, go through all inits tensors and find which of its
    // dimensions have been undef padded. The corresponding dimensions of the
    // operands can then also be undef padded.
    // If true for every dim of the operand, for every output, the entire
    // operand can be undef padded.
    for (auto [operandMap, operand] :
         llvm::zip_equal(indexingMaps, linalgOp->getOpOperands())) {
      auto operandPadOp = operand.get().getDefiningOp<tensor::PadOp>();
      // Already undef padded.
      if (!operandPadOp || hasUndefPadding(operandPadOp))
        continue;

      bool canUndefPad = true;
      for (OpOperand &inits : linalgOp.getDpsInitsMutable()) {
        auto padOp = inits.get().getDefiningOp<tensor::PadOp>();
        if (!padOp || !hasUndefPadding(padOp)) {
          canUndefPad = false;
          break;
        }

        // Initially, all padded dims need non-undef padding.
        llvm::SmallBitVector needNonUndefPadding = operandPadOp.getPaddedDims();

        llvm::SmallBitVector initsPaddedDims = padOp.getPaddedDims();
        // Create a mapping from the init dims to the operand dims.
        // The init dims affine map must be invertible for this to be possible.
        AffineMap initsMap = linalgOp.getMatchingIndexingMap(&inits);
        AffineMap inverted = inverseAndBroadcastProjectedPermutation(initsMap);
        if (!inverted) {
          canUndefPad = false;
          break;
        }

        AffineMap initsToOperand = operandMap.compose(inverted);
        for (unsigned index : initsPaddedDims.set_bits())
          if (std::optional<unsigned> position =
                  initsToOperand.getResultPosition(
                      rewriter.getAffineDimExpr(index)))
            needNonUndefPadding[*position] = false;

        if (needNonUndefPadding.any()) {
          canUndefPad = false;
          break;
        }
      }
      if (!canUndefPad)
        continue;

      rewriter.modifyOpInPlace(linalgOp, [&, &operand = operand] {
        operand.set(cloneWithUndefPad(rewriter, operandPadOp));
      });
      changed = true;
    }
    return success(changed);
  }
};

} // namespace

namespace {

/// Expands a 'pad_undef(empty)' to a larger empty.
struct ExpandPaddedEmptyOp : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!hasUndefPadding(padOp))
      return failure();

    auto emptyOp = padOp.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return failure();

    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyResultShapes(rewriter, padOp, dims)))
      return failure();

    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        padOp, dims.front(), getElementTypeOrSelf(padOp.getType()));
    return success();
  }
};

/// Expands a 'pad_undef(extract_slice)' to a larger extract_slice if possible.
struct ExpandPaddedSlice : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Lower padding and dynamic padding values are currently unsupported.
    if (!hasUndefPadding(padOp) || !padOp.hasZeroLowPad() ||
        !padOp.getHigh().empty())
      return failure();

    auto extractSlice =
        padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    // Only zero-offset supported right now to simplify the logic.
    if (!extractSlice || !extractSlice.hasZeroOffset())
      return failure();

    TypedValue<RankedTensorType> source = extractSlice.getSource();
    if (extractSlice.getType().getRank() != source.getType().getRank())
      return failure();

    // Check that the pad does not make the tensor larger than the original
    // source of the slice.
    ArrayRef<int64_t> high = padOp.getStaticHigh();
    for (auto [highPad, sourceShape, resultShape] : llvm::zip_equal(
             high, source.getType().getShape(), padOp.getType().getShape())) {
      if (highPad == 0)
        continue;

      if (sourceShape < resultShape)
        return failure();
    }

    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyResultShapes(rewriter, padOp, dims)))
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        padOp, source, /*offsets=*/
        SmallVector<OpFoldResult>(source.getType().getRank(),
                                  rewriter.getIndexAttr(0)),
        /*sizes=*/dims.front(),
        /*strides=*/
        SmallVector<OpFoldResult>(source.getType().getRank(),
                                  rewriter.getIndexAttr(1)));
    return success();
  }
};
} // namespace

void PadToTilingConfig::runOnOperation() {
  // TODO: This is seems like a horrible restriction and should be fixed.
  if (getOperation()
          ->walk([&](tensor::PadOp padOp) {
            padOp.emitError(
                "pass does not handle pre-existing padding operations");
            return WalkResult::interrupt();
          })
          .wasInterrupted())
    return signalPassFailure();

  SmallVector<linalg::LinalgOp> workList;
  getOperation()->walk([&](linalg::LinalgOp linalgOp) {
    if (!canZeroPad(linalgOp))
      return;
    workList.push_back(linalgOp);
  });

  std::optional<IntegerAttr> attr = getConfigIntegerAttr(
      IREE::HAL::ExecutableTargetAttr::lookup(getOperation()), "compute_cores");

  // getOperation()->emitWarning() << "\nvvv before padding vvv";
  //  Pad every linalg op to a multiple of all applied tile sizes.
  for (linalg::LinalgOp &linalgOp : workList)
    if (failed(padToTileSize(linalgOp, attr)))
      return signalPassFailure();
  getOperation()->emitWarning() << "\nvvv after padding vvv";

  // Remove redundant buffer inserted by padding row dimension
  {
    RewritePatternSet patterns(&getContext());
    auto config = mlir::GreedyRewriteConfig();
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    patterns.insert<RemoveRedundantBuffer>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      getOperation()->emitWarning()
          << "\nremoving redunand buffer pattern failed\n";
      return signalPassFailure();
    }
  }
  getOperation()->emitWarning() << "\nvvv after pattern vvv";
  // First perform just the conversion of zero-pads to undef-pads.
  // These must run separately from later patterns that may erase pad ops
  // entirely which discards information required by these patterns.
  {
    RewritePatternSet patterns(&getContext());
    patterns.insert<OptimizeElementwisePad, PropagateUnusedOutputPads,
                    OptimizeContractionOutputPad>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

  {
    RewritePatternSet patterns(&getContext());
    linalg::populateSwapExtractSliceWithFillPatterns(patterns);
    linalg::FillOp::getCanonicalizationPatterns(patterns, &getContext());
    getContext()
        .getLoadedDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    tensor::PadOp::getCanonicalizationPatterns(patterns, &getContext());
    patterns.insert<ExpandPaddedEmptyOp, ExpandPaddedSlice>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
}
