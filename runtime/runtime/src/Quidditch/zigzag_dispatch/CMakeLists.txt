iree_cc_library(
    NAME
    zigzag_dispatch
    SRCS
    zigzag_dispatch.c
    DEPS
    snRuntimeInterface
    # notice there is no iree::base included as a dependency
    PUBLIC
)