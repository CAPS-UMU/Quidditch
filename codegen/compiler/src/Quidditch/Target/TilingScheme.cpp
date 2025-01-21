#include "TilingScheme.h"

// using namespace quidditch;
using namespace mlir;
using namespace mlir::iree_compiler;

// Tiling Scheme Functions defined below
namespace quidditch {

TileInfoTbl *fillTileInfoTable(TileInfoTbl *tbl, const std::string &filePath,
                               std::string &errs) {
  TileInfoTbl *result = tbl;
  // try to open file
  std::ifstream ifs(filePath);
  if (!ifs.is_open()) {
    std::stringstream ss;
    ss << "\nTiling Scheme File does not exist or cannot be opened.\n"
       << "Troublesome file path is " << filePath << "\n";
    errs = ss.str();
    return 0;
  }
  // try to read file
  std::stringstream ss;
  ss << ifs.rdbuf();
  if (ss.str().length() == 0) {
    errs = "\nTiling Scheme file cannot have content length of 0\n";
    ifs.close();
    return 0;
  }
  // try to parse list of schemes
  if (!parseTilingSchemes(tbl, StringRef(ss.str()), errs)) {
    result = 0;
  }
  ifs.close();
  return result;
}

TileInfoTbl *exportTileInfoTable(TileInfoTbl *tbl, const std::string &filePath,
                                 std::string &errs) {
  if (tbl == 0) {
    return 0;
  }
  TileInfoTbl *result = tbl;
  std::stringstream outputFilePath;
  outputFilePath << filePath << "-exported.json";
  // try to open file
  std::ofstream ofs(outputFilePath.str(), std::ofstream::out);
  if (!ofs.is_open()) {
    std::stringstream ss;
    ss << "\nTiling Scheme File does not exist or cannot be opened.\n"
       << "Troublesome file path is " << outputFilePath.str() << "\n";
    errs = ss.str();
    return 0;
  }
  ofs << "yodelayheehoooooo~~~~~!\n";
  // std::pair< iterator, bool > 	insert (KV E)

  // ofs << tester;
  // llvm::json::Value::Value 	( 	const std::map< std::string, Elt > & C
  // ) llvm::json::Value::Value 	( 	const llvm::SmallVectorImpl<
  // char > &  	V	) std::vector<int> myrtleCost;
  //  std::map< std::string, llvm::json::Value> costMap = {};
  std::map<std::string, int> costMap = {};
  // try to write to file
  /*
  struct Object::KV {
  ObjectKey K;
  Value V;
};
  llvm::json::Value toJSON(const Position &P) {
  return llvm::json::Object{
      {"line", P.line},
      {"character", P.character},
  };
}
  */
  for (const auto &pear : *tbl) {
    ofs << pear.first << "\n";
    costMap.insert(std::pair<std::string, int>(pear.first, 7));
  }
  // auto hoodle = llvm::json::Object(costMap);
  // auto costMapAsJson = llvm::json::Value({{"hoodle","yodel"},{"yohoho, 5"}});
  std::string blank = "";
  llvm::raw_string_ostream ros = llvm::raw_string_ostream(blank);
  //ros << llvm::json::toJSON(costMap);
  // auto tester = llvm::json::Object();
  // tester.insert({"name",5});
  // ros << tester;//llvm::json::Value(tester);
  // ofs << ros.str();
  llvm::json::OStream J(ros);
    J.array([&] {
    for (const auto &pear : *tbl)
      J.object([&] {
        J.attribute(pear.first, int64_t(4));
        J.attributeArray("myrtleCost", [&] {
          for (int64_t cost : pear.second.myrtleCost)
            J.value(cost);
        });
      });
  });
   ofs << ros.str();
  /*
  J.array([&] {
    for (const Event &E : Events)
      J.object([&] {
        J.attribute("timestamp", int64_t(E.Time));
        J.attributeArray("participants", [&] {
          for (const Participant &P : E.Participants)
            J.value(P.toString());
        });
      });
  });
  */
 // [ { "timestamp": 19287398741, "participants": [ "King Kong", "Miley Cyrus", "Cleopatra" ] }, ... ]
  // std::stringstream ss;
  // ss << ifs.rdbuf();
  // if (ss.str().length() == 0) {
  //   errs = "\nTiling Scheme file cannot have content length of 0\n";
  //   ifs.close();
  //   return 0;
  // }
  // // try to parse list of schemes
  // if (!parseTilingSchemes(tbl, StringRef(ss.str()), errs)) {
  //   result = 0;
  // }
  ofs.close();
  return result;
}

bool parseTilingSchemes(TileInfoTbl *tbl, llvm::StringRef fileContent,
                        std::string &errs) {
  // try to parse
  llvm::Expected<llvm::json::Value> maybeParsed =
      llvm::json::parse(fileContent);
  if (!maybeParsed) {
    std::stringstream ss;
    ss << "\nError when parsing JSON file contents: "
       << llvm::toString(maybeParsed.takeError()) << "\n";
    errs = ss.str();
    return false;
  }
  // try to get the top level json object
  if (!maybeParsed->getAsObject()) {
    errs = "\nError: top-level value is not a JSON object\n";
    return false;
  }
  llvm::json::Object *O = maybeParsed->getAsObject();
  // make sure object has at least one field
  if (O->empty()) {
    errs = "\nError: top-level JSON object is empty\n";
    return false;
  }
  // try to parse each tiling scheme from function name key
  std::stringstream ss;
  for (const auto &func : *O) {
    struct TilingScheme ts = parseTilingScheme(func.getSecond(), errs);
    if (!ts.valid) {
      return false;
    } else {
      tbl->insert(std::pair(func.getFirst().str(), ts));
      ss << func.getFirst().str() << ":\n";
      ss << ts;
    }
  }
  errs = ss.str();
  return true;
}

struct TilingScheme parseTilingScheme(llvm::json::Value v, std::string &errs) {
  struct TilingScheme ts;
  auto O = v.getAsObject();
  if (!O) {
    errs = "RHS of key_value pair is not a JSON object!";
    return ts;
  }
  bool read_tile_sizes = parseListOfListOfInts(O, "tile-sizes", ts.tiles, errs);
  bool read_loop_order = parseListOfListOfInts(O, "loop-order", ts.order, errs);
  bool read_dual_buffer = parseBool(O, "dual-buffer", ts.dualBuffer, errs);
  ts.valid = read_tile_sizes && read_loop_order && read_dual_buffer;
  return ts;
}

// TODO: call parseListOfInts inside parseListOfListOfInts
bool parseListOfListOfInts(llvm::json::Object *obj, std::string listName,
                           std::vector<std::vector<int>> &out,
                           std::string &errs) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) {
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' does not exist \n ";
    errs = ss.str();
    return false;
  }

  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' is not a JSON array \n ";
    errs = ss.str();
    return false;
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  for (const auto &Item :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!Item.getAsArray()) {
      std::stringstream ss;
      ss << "\nError: elt of '" << listName << "' is not also a JSON array \n ";
      errs = ss.str();
      return false;
    }
    std::vector<int> sublist;
    int bound;
    for (const auto &elt :
         *(Item.getAsArray())) { // loop over a json::Array type
      if (!fromJSON(elt, bound, Root)) {
        std::stringstream ss;
        ss << llvm::toString(Root.getError()) << "\n";
        errs = ss.str();
        return false;
      }
      sublist.push_back(bound);
    }
    out.push_back(sublist);
  }
  return true;
}

bool parseListOfInts(llvm::json::Object *obj, std::string listName,
                     std::vector<int> &out, std::string &errs) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) {
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' does not exist \n ";
    errs = ss.str();
    return false;
  }
  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' is not a JSON array \n ";
    errs = ss.str();
    return false;
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  int theNumber;
  for (const auto &elt :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!fromJSON(elt, theNumber, Root)) {
      std::stringstream ss;
      ss << llvm::toString(Root.getError()) << "\n";
      errs = ss.str();
      return false;
    }
    out.push_back(theNumber);
  }
  return true;
}

bool parseBool(llvm::json::Object *obj, std::string boolName, bool &out,
               std::string &errs) {
  llvm::json::Value *theBool = obj->get(StringRef(boolName));
  if (!theBool) {
    std::stringstream ss;
    ss << "\nError: field labeled '" << boolName << "' does not exist \n ";
    errs = ss.str();
    return false;
  }
  if (!theBool->getAsBoolean()) { // getAsBoolean returns an optional boolean
    std::stringstream ss;
    ss << "\nError: field labeled '" << boolName << "' is not a boolean \n ";
    errs = ss.str();
    return false;
  }
  out = *(theBool->getAsBoolean());
  return true;
}

// bool TilingScheme::exportWorkloadsToFile() {
//   if (workloadFileName.compare("") == 0) {
//     updateErrs("\nCannot export to nameless workload file.\n");
//     return false;
//   }
//   std::fstream fs;
//   fs.open(workloadFileName,
//           std::fstream::in | std::fstream::out | std::fstream::app);
//   fs << workloads;
//   fs.close();
//   return true;
// }

bool TilingScheme::getMyrtleCost(llvm::SmallVector<int64_t> &out) {
  if (out.size() != myrtleCost.size()) {
    return false;
  } else {
    for (size_t i = 0; i < tiles.size(); i++) {
      out[i] = (int64_t)myrtleCost[i];
    }
  }
  return true;
}

bool TilingScheme::getTiles_flat(llvm::SmallVector<int64_t> &out) {
  if (out.size() != tiles.size()) {
    return false;
  } else {
    for (size_t i = 0; i < tiles.size(); i++) {
      out[i] = (int64_t)tiles[i][0];
    }
  }
  return true;
}

bool TilingScheme::getOrder_flat(llvm::SmallVector<int64_t> &out) {
  if (out.size() != order.size()) {
    return false;
  } else {
    for (size_t i = 0; i < order.size(); i++) {
      out[i] = (int64_t)order[i][0];
    }
  }
  return true;
}

std::string TilingScheme::str() {
  std::stringstream ts_ss;
  ts_ss << *this;
  return ts_ss.str();
}

std::stringstream &operator<<(std::stringstream &ss,
                              const struct TilingScheme &ts) {
  ss << "tiling scheme: {\nbounds: [ ";
  for (const auto &sublist : ts.tiles) {
    ss << "[ ";
    for (const auto &tile : sublist) {
      ss << " " << tile << " ";
    }
    ss << "] ";
  }
  ss << "]\n";
  ss << "order: [ ";
  for (const auto &sublist : ts.order) {
    ss << "[ ";
    for (const auto &pos : sublist) {
      ss << " " << pos << " ";
    }
    ss << "] ";
  }
  ss << "]\n";
  ss << "dual buffer: " << ts.dualBuffer << "\n";
  ss << "myrtle cost: [ ";
  for (const auto &cost : ts.myrtleCost) {
    ss << " " << cost << " ";
  }
  ss << "]\n}";
  return ss;
}

} // namespace quidditch
