quidditch_module(SRC pamplemousse.mlir ASSERT_XDSL FLAGS --iree-quidditch-zigzag-tiling-scheme="pomelo-hoodle.json")

#quidditch_module(SRC pamplemousse.mlir LLVM FLAGS --iree-quidditch-zigzag-tiling-scheme="hoodle.json")


add_executable(pomelo main.c)
target_link_libraries(
    pomelo
    PRIVATE
    samples_util
    pamplemousse
    snRuntime
    Quidditch::dispatch::dispatch
)
