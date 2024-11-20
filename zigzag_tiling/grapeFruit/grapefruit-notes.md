# GrapeFruit test case notes

[back to landing](../README.md)

- GrapeFruit is a copy of the NsNet2 test case.
- The only difference between GrapeFruit and NsNet2 is that when the `ConfigureUsingZigZag` pass runs, the kernel `main$async_dispatch_8_matmul_transpose_b_1x600x600_f64` gets tiled with ZigZag tile sizes.
- Full Quidditch Pass Pipeline defined in [this file](../../codegen/compiler/src/Quidditch/Target/QuidditchTarget.cpp)
- `ConfigureUsingZigZag` pass defined [here](../../codegen/compiler/src/Quidditch/Target/ConfigureUsingZigzag.cpp)

## Building + Running

```
cd build
```

Configure:

```
cmake .. -GNinja \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
```

Build:

```
ninja -j 20
```

Run:

```
../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit
```

