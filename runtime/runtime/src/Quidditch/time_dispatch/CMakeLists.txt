
iree_cc_library(
    NAME
    time_dispatch
    SRCS
    time_dispatch.c
    DEPS
    snRuntimeInterface
    iree::base
    PUBLIC
)
