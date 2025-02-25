#!/bin/sh

tests="tiledMatmul6.mlir \
tiledMatmul12.mlir \
"
for i in $tests
do
   echo "Compiling $i for riscv..."
   sh compile-for-riscv.sh "$i"
done
