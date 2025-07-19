quidditch_module(SRC persimmon.mlir ASSERT_XDSL)
#quidditch_module(SRC persimmon.mlir LLVM)


add_executable(caqui main.c)
target_link_libraries(
    caqui
    PRIVATE
    samples_util
    persimmon
    snRuntime
    Quidditch::dispatch::dispatch
)
