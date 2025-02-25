#!/bin/sh
# Run this script from inside the runtime/tests directory
here=$(pwd)
basename=`basename $1 | sed 's/[.][^.]*$//'`
echo "$basename"
echo "${basename^}"


# compile the mlir kernel
sh compile-for-riscv.sh "$basename" &&\

## compile c code with mlir kernel linked in as object file
cd ../../build &&\
cmake .. -GNinja \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &&\
ninja -j 20

# run the program
../../toolchain/bin/snitch_cluster.vlt runtime/tests/${basename^} &&\

# return to runtime/tests directory
cd "$here"