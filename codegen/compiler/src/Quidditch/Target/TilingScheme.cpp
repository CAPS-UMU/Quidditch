#include "TilingScheme.h"

// using namespace quidditch;
using namespace mlir;
using namespace mlir::iree_compiler;

// Tiling Scheme Functions defined below
namespace quidditch {

TileInfoTbl* fillTileInfoTable(TileInfoTbl* tbl, std::string filePath){
  TileInfoTbl* result = tbl;
  return result;
}

void TilingScheme::setTotalLoopCount() {
  unsigned total = 0;
  for (const auto &bound : bounds) {
    total += (bound.size() +
              1); // for each loop getting tiled, count the extra affine loop
                  // needed to calculate the first level indexing inside a tile
  }
  //   LLVM_DEBUG(
  //       llvm::dbgs() << "[" DEBUG_TYPE
  //                       "] total number of loops in tiled loop nest will be "
  //                    << total << " \n");
  totalLoopCount = total;
}

void TilingScheme::buildFinalIndices() {
  // std::vector<std::vector<int>> bounds;
  // finalIndices
  for (size_t i = 0; i < bounds.size(); i++) {
    finalIndices.push_back(std::vector<int>());
    for (size_t j = 0; j < bounds[i].size(); j++) {
      size_t finalIndex = totalLoopCount - findSubloop(i, j) - 1;
      finalIndices[i].push_back(finalIndex);
    }
  }
}

int TilingScheme::findSubloop(size_t i, size_t j) {
  for (size_t k = 0; k < order.size(); k++) {
    if (((size_t)order[k][0] == i) && ((size_t)order[k][1] == j)) {
      return k;
    }
  }
  //   LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
  //                              "] Error: Could not find subloop in tiling
  //                              scheme " "order. Returning negative index...
  //                              \n");
  return -1;
}

bool TilingScheme::exportWorkloadsToFile() {
  if (workloadFileName.compare("") == 0) {
    updateErrs("\nCannot export to nameless workload file.\n");
    return false;
  }
  std::fstream fs;
  fs.open(workloadFileName,
          std::fstream::in | std::fstream::out | std::fstream::app);
  fs << workloads;
  fs.close();
  return true;
}

void TilingScheme::initialize(std::string filename, std::string filename2) {
  valid = true;
  workloadFileName = filename2;
  //exportWorkloadsToFile();
  // try to open file
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    updateErrs("\nTiling Scheme File does not exist or cannot be opened.\n");
    valid = false;
    return;
  }
  // try to read file
  std::stringstream ss;
  ss << ifs.rdbuf();
  if (ss.str().length() == 0) {
    updateErrs("\nTiling Scheme file cannot have content length of 0\n");
    ifs.close();
    valid = false;
    return;
  }
  //  try to parse file contents
  parseTilingScheme(StringRef(ss.str()));
  ifs.close();
}

std::string TilingScheme::str() {
  std::stringstream ts_ss;
  ts_ss << *this;
  return ts_ss.str();
}

// helpers for processing tiling scheme input
void TilingScheme::parseListOfListOfInts(llvm::json::Object *obj,
                                         std::string listName,
                                         std::vector<std::vector<int>> &out) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) { // getAsArray returns a (const json::Array *)
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' does not exist \n ";
    updateErrs(ss.str());
    valid = false;
    return;
  }

  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' is not a JSON array \n ";
    updateErrs(ss.str());
    valid = false;
    return;
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  for (const auto &Item :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!Item.getAsArray()) {
      std::stringstream ss;
      ss << "\nError: elt of '" << listName << "' is not also a JSON array \n ";
      updateErrs(ss.str());
      valid = false;
      return;
    }
    std::vector<int> sublist;
    int bound;
    for (const auto &elt :
         *(Item.getAsArray())) { // loop over a json::Array type
      if (!fromJSON(elt, bound, Root)) {
        std::stringstream ss;
        ss << llvm::toString(Root.getError()) << "\n";
        updateErrs(ss.str());
        valid = false;
        return;
      }
      sublist.push_back(bound);
    }
    out.push_back(sublist);
  }
}

void TilingScheme::parseTilingScheme(llvm::StringRef fileContent) {
  llvm::Expected<llvm::json::Value> maybeParsed =
      llvm::json::parse(fileContent);
  if (!maybeParsed) {
    std::stringstream ss;
    ss << "\nError when parsing JSON file contents: "
       << llvm::toString(maybeParsed.takeError()) << "\n";
    updateErrs(ss.str());
    valid = false;
    return;
  }
  // try to get the top level json object
  if (!maybeParsed->getAsObject()) {
    updateErrs("\nError: top-level value is not a JSON object\n");
    valid = false;
    return;
  }
  llvm::json::Object *O = maybeParsed->getAsObject();
  // try to read the two fields
  parseListOfListOfInts(O, "bounds", bounds);
  parseListOfListOfInts(O, "order", order);
}

std::stringstream &operator<<(std::stringstream &ss,
                              const struct TilingScheme &ts) {
  ss << "tiling scheme: {\nbounds: [ ";
  for (const auto &sublist : ts.bounds) {
    ss << "[ ";
    for (const auto &bound : sublist) {
      ss << " " << bound << " ";
    }
    ss << "] ";
  }
  ss << "]\n";
  ss << "finalIndices: [ ";
  for (const auto &sublist : ts.finalIndices) {
    ss << "[ ";
    for (const auto &pos : sublist) {
      ss << " " << pos << " ";
    }
    ss << "] ";
  }
  ss << "]\n}";
  ss << "order: [ ";
  for (const auto &sublist : ts.order) {
    ss << "[ ";
    for (const auto &pos : sublist) {
      ss << " " << pos << " ";
    }
    ss << "] ";
  }
  ss << "]\n}";
  return ss;
}

// there must be a better way to do this, but for now, let's get the elephant
// dancing.
void TilingScheme::updateErrs(std::string err) {
  std::stringstream ss(errs);
  ss << err;
  errs = ss.str();
}
void TilingScheme::updateWorkloads(std::string wrkload) {
  std::stringstream ss(workloads);
  ss << wrkload;
  workloads = ss.str();
}

} // namespace quidditch
