#!/bin/sh
# Run this script using "dot source": `. setup.sh`

#llvmMlirBuild='/home/hoppip/llvm-project-17/build-riscv/bin' #for mlir-opt-17
llvmMlirBuild='/home/hoppip/llvm-project-pistachio/build-riscv/bin' # modify this
riscvToolChainInstall='/home/hoppip/riscv/bin'                      # modify this
spikeBuild='/home/hoppip/riscv-isa-sim/build'                       # modify this

# paths from my laptop :)
# llvmMlirBuild='/home/emily/llvm-project-pistachio/build-riscv/bin' # modify this
# riscvToolChainInstall='/opt/riscv/bin'                             # modify this
# spikeBuild='/home/emily/riscv-isa-sim/build'                       # modify this

# add your LLLVM MLIR build path to your path
export PATH="$llvmMlirBuild:$PATH"

# add your risc-v toolchain install path to your path
export PATH="$riscvToolChainInstall:$PATH"

# set the SPIKE environment variable to your spike build directory
export SPIKE="$spikeBuild"