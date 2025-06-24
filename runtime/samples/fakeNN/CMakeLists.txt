iree_turbine(SRC fakeNN.py DST ${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc DTYPE "f64" M 1 N 120 K 40)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc DST fakeNN FLAGS --mlir-disable-threading --iree-quidditch-export-costs=/home/hoppip/Quidditch/runtime/samples/fakeNN/fakeNN-tile-sizes-costs.txt --iree-quidditch-import-tiles=/home/hoppip/Quidditch/runtime/samples/fakeNN/fakeNN-tile-sizes.json)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc LLVM DST fakeNN_llvm FLAGS --iree-quidditch-export-costs=/home/hoppip/Quidditch/runtime/samples/fakeNN/fakeNN-tile-sizes-costs.txt --iree-quidditch-import-tiles=/home/hoppip/Quidditch/runtime/samples/fakeNN/fakeNN-tile-sizes.json)
add_library(fakeNN_util fakeNN_util.c)
target_link_libraries(fakeNN_util
    PRIVATE
    samples_util
    snRuntimeInterface
    Quidditch::dispatch::dispatch
)
target_include_directories(fakeNN_util INTERFACE ${CMAKE_CURRENT_LIST_DIR})

macro(create_experiment_variant)
  cmake_parse_arguments(_RULE "PRECOMMIT;NIGHTLY" "TARGET;IREE_MODULE;QUERY_FUNC" "" ${ARGN})

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c "\
#include <${_RULE_IREE_MODULE}.h>

#include \"fakeNN_util.h\"

int main() {
  return run_fakeNN_experiment(${_RULE_QUERY_FUNC});
}
")
  add_executable(${_RULE_TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c)
  target_link_libraries(
      ${_RULE_TARGET}
      PRIVATE
      fakeNN_util
      ${_RULE_IREE_MODULE}
      snRuntime
  )
endmacro()

create_experiment_variant(
    TARGET FakeNN
    IREE_MODULE fakeNN
    QUERY_FUNC "quidditch_compiled_fake_n_n_linked_quidditch_library_query"
)
create_experiment_variant(
    TARGET FakeNNLLVM
    IREE_MODULE fakeNN_llvm
    QUERY_FUNC "compiled_fake_n_n_linked_llvm_cpu_library_query"
)