

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

namespace quidditch{
struct TilingScheme {
  bool valid = false;
  uint64_t totalLoopCount = 0;
  std::vector<std::vector<int>> bounds;
  std::vector<std::vector<int>> order;
  std::vector<std::vector<int>> finalIndices;
  TilingScheme() = default;
  void initialize(std::string filename, std::string filename2 = "");
  std::string str();
  std::string errs = "";
  std::string workloads = "DONKEY";
  std::string workloadFileName = "";
  bool exportWorkloadsToFile();
  friend std::stringstream &operator<<(std::stringstream &ss,
                                       const struct TilingScheme &ts);
  int findSubloop(size_t i, size_t j);
  void setTotalLoopCount();
  void buildFinalIndices();
  void parseTilingScheme(llvm::StringRef fileContent);
  void parseListOfListOfInts(llvm::json::Object *obj, std::string listName,
                             std::vector<std::vector<int>> &out);
void updateErrs(std::string err);
void updateWorkloads(std::string wrkload);
};

//  std::unordered_map<std::string, std::string> upgjkgh;
typedef std::unordered_map<std::string, struct quidditch::TilingScheme> TileInfoTbl;



TileInfoTbl* fillTileInfoTable(TileInfoTbl* tbl, std::string filePath);



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