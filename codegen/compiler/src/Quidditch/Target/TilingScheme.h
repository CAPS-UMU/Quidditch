

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

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
// #include "mlir/IR/BuiltinTypes.h"

namespace quidditch{
struct TilingScheme {
  bool valid = false;
  std::vector<std::vector<int>> tiles;
  std::vector<std::vector<int>> order;
  std::vector<int> myrtleCost;
  bool dualBuffer = false;
  std::string errs = "";
  // member funcs
  TilingScheme() = default;
  std::string str();
  bool getTiles_flat( llvm::SmallVector<int64_t>& out);
  bool getOrder_flat( llvm::SmallVector<int64_t>& out);
  bool getDualBuffer(){return dualBuffer;}
  // overloaded output operator
  friend std::stringstream &operator<<(std::stringstream &ss,
                                       const struct TilingScheme &ts);
};

//  std::unordered_map<std::string, std::string> upgjkgh;
typedef std::unordered_map<std::string, struct quidditch::TilingScheme> TileInfoTbl;

bool parseTilingSchemes(TileInfoTbl* tbl, llvm::StringRef fileContent, std::string& errs);
struct TilingScheme parseTilingScheme(llvm::json::Value v, std::string& errs);
TileInfoTbl* fillTileInfoTable(TileInfoTbl* tbl, const std::string& filePath, std::string& errs);
bool parseListOfListOfInts(llvm::json::Object *obj,
                                         std::string listName,
                                         std::vector<std::vector<int>> &out, std::string& errs);
bool parseListOfInts(llvm::json::Object *obj,
                                         std::string listName,
                                         std::vector<int> &out, std::string& errs);
bool parseBool(llvm::json::Object *obj,
                                         std::string listName,
                                          bool& out, std::string& errs);

/*
hashtable from string --> tilingScheme object
tileConfig Object supports
- printing itself out
- exporting its tile-sizes and interchange fields as Small<Int>

minimal test:
have QuidditchTarget.cpp creates empty hashtable. 
Then, tries to open the json file,  
- if this is possible, continue with configureTiling pass. 
- If filepath is empry, SKIP tiling pass with WARNING.
- if file path is NOT empty and DOES NOT EXIST, set invalid to true, and have configure tiling pass fail inside its constructor
*/
}