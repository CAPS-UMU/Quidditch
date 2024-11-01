quidditch_module(SRC pumpkin.mlir ASSERT_XDSL FLAGS --iree-quidditch-zigzag-tiling-scheme=hoodle.json --iree-quidditch-output-tiled=true)

add_executable(calabaza main.c)
target_link_libraries(
    calabaza
    PRIVATE
    samples_util
    pumpkin
    snRuntime
    Quidditch::dispatch::dispatch
)
