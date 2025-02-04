iree_turbine(SRC grapeFruit.py DST ${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc DTYPE "f64")
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc DST grapeFruit FLAGS --mlir-disable-threading --iree-quidditch-export-costs=/home/hoppip/Quidditch/comparing-tile-sizes/0-40-200/tilingCosts.json --iree-quidditch-import-tiles=/home/hoppip/Quidditch/comparing-tile-sizes/tile-sizes-to-test/0-40-200.json)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc LLVM DST grapeFruit_llvm FLAGS --iree-quidditch-export-costs=/home/hoppip/Quidditch/comparing-tile-sizes/0-40-200/tilingCosts.json --iree-quidditch-import-tiles=/home/hoppip/Quidditch/comparing-tile-sizes/tile-sizes-to-test/0-40-200.json)
add_library(grapeFruit_util grapeFruit_util.c)
target_link_libraries(grapeFruit_util
    PRIVATE
    samples_util
    snRuntimeInterface
    Quidditch::dispatch::dispatch
)
target_include_directories(grapeFruit_util INTERFACE ${CMAKE_CURRENT_LIST_DIR})

macro(create_experiment_variant)
  cmake_parse_arguments(_RULE "PRECOMMIT;NIGHTLY" "TARGET;IREE_MODULE;QUERY_FUNC" "" ${ARGN})

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c "\
#include <${_RULE_IREE_MODULE}.h>

#include \"grapeFruit_util.h\"

int main() {
  return run_grapeFruit_experiment(${_RULE_QUERY_FUNC});
}
")
  add_executable(${_RULE_TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c)
  target_link_libraries(
      ${_RULE_TARGET}
      PRIVATE
      grapeFruit_util
      ${_RULE_IREE_MODULE}
      snRuntime
  )
endmacro()

create_experiment_variant(
    TARGET GrapeFruit
    IREE_MODULE grapeFruit
    QUERY_FUNC "quidditch_compiled_ns_net2_linked_quidditch_library_query"
)
create_experiment_variant(
    TARGET GrapeFruitLLVM
    IREE_MODULE grapeFruit_llvm
    QUERY_FUNC "compiled_ns_net2_linked_llvm_cpu_library_query"
)