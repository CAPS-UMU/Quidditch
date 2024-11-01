quidditch_module(SRC pamplemousse.mlir ASSERT_XDSL FLAGS --iree-quidditch-zigzag-tiling-scheme="hoodle.json" --iree-quidditch-output-tiled=true)

add_executable(pomelo main.c)
target_link_libraries(
    pomelo
    PRIVATE
    samples_util
    pamplemousse
    snRuntime
    Quidditch::dispatch::dispatch
)
