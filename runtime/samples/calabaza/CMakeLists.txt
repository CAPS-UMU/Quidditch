set( MYRTLE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../myrtle/myrtle/myrtle.py)
quidditch_module(SRC pumpkin.mlir ASSERT_XDSL FLAGS --iree-quidditch-myrtle=${MYRTLE_PATH} --iree-quidditch-myrtle-mode=svrcyc --iree-quidditch-myrtle-out=${CMAKE_CURRENT_LIST_DIR}/ts-answer.json) 
add_executable(calabaza main.c)
target_link_libraries(
    calabaza
    PRIVATE
    samples_util
    pumpkin
    snRuntime
    Quidditch::dispatch::dispatch
)
