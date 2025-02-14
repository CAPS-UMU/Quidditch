# Myrtle Auto-Tiling

```
[hoppip@inf-205-141 grapferuit-runs]$ cd comparing-tile-sizes/
[hoppip@inf-205-141 comparing-tile-sizes]$ cd 0-80-80/
[hoppip@inf-205-141 0-80-80]$ grep run_output.txt "dispatch 1"
grep: dispatch 1: No such file or directory
[hoppip@inf-205-141 0-80-80]$ grep "dispatch 1" run_output.txt
dispatch 1: 1109900 - 958493 = 151407
[hoppip@inf-205-141 0-80-80]$ grep "cycles" run_output.txt
cycles 1355840
```

Notes:

```
tensor<1x400xf64>, tensor<1200x400xf64> -> tensor<1x1200xf64>
                           ^
                           must be divisible by 8
                           |
```

Heuristics for Picking Tiles

- dimension 2 (1200 in the above example) must be divisible by 8
- we want to make the body of a hardware loop at least 4 instructions (or will get pipeline bubbles)

| Hardware Loop Body FMADD >= 4 | Tile Size also divisible by 8 |
| ----------------------------- | ----------------------------- |
| 3                             | 24                            |
| 4                             | 32                            |
| 5                             | 40                            |
| 6                             | 48                            |
| 7                             | 56                            |
| 8                             | 64                            |
| 9                             | 72                            |
| 10                            | 80                            |
| 11                            | 88                            |
| 12                            | 96                            |
| 13                            | 104                           |
| 14                            | 112                           |

**0-72-50 ERROR buy why???** Doesn't fit!!

Trying different tile sizes results:

```
0-24-100: (24, 200, 105915);  1177816 cycles for full NN
0-24-120: (24, 120, 167326);  1420971 cycles for full NN
0-32-100: (32, 100, 90218);   1112995 cycles for full NN
0-40-100: (40, 100, 89788);   1112529 cycles for full NN
0-48-80: (48, 80, 106705);    1176774 cycles for full NN
0-40-80: (40, 80, 107211);    1181547 cycles for full NN
0-80-40: (80, 40, 106171);    1174862 cycles for full NN
0-80-50: (80, 50, 108982);    1185914 cycles for full NN
0-56-50: (56, 50, 109661);    1189511 cycles for full NN
0-64-50: (64, 50, 116837);    1220523 cycles for full NN
```

What about some super small reduction dimensions like 10 or 20? Or even 100 x 40 instead of the other way around?

like 80 x 40 and 40 x 80? which is better?

## How to Test Myrtle with Different Tile Sizes

1. Navigate to the comparing-tile-sizes directory
   ```
   cd comparing-tile-sizes
   ```

2. Add tiling schemes (as json files) to the folder `tile-sizes-to-test`
### One Tiling Scheme

3. cmake and ninja build only one of the tiling schemes inside `tile-sizes-to-test` :
```
sh manyGrapefruits.sh actuallyOnlyOne 0-40-100.json noRun
```

4.  Run only one of the tiling schemes using verilator:

```
sh manyGrapefruits.sh actuallyOnlyOne 0-40-100.json
```

### Many Tiling Schemes

3. Run cmake and ninja build for all tiling schemes inside `tile-sizes-to-test` with
   ```
   sh manyGrapefruits.sh cmake
   ```

4. Run each tiled nsnet one by one with
   ```
   sh manyGrapefruits.sh
   ```

5. Results will be in subfolders named after the json file specifying the tiling scheme. For example, a directory structure like
   ```
   comparing-tile-sizes/
   |---- 0-48-100/
         |---- logs/
         |---- buildOutput.txt
         |---- GrapeFruit
         |---- tilingCosts.json
   ```

## 0. Daily Use Commands

Inside `Quidditch` directory, do

```
source ./venv/bin/activate
```
Navigate to your build directory.

If you made changes to a `CMakeLists.txt` file, rerun cmake with
```
cmake .. -GNinja \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
```

After making changes to quidditch, rebuild with 

```
clear;ninja -j 20
```

 ## I. Run a test case

##### A. Directly Invoking Verilator

Navigate to `<build-directory>` then do

```
../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit
```

```
../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/caqui/caqui
```

```
../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/vec_multiply/vec_multiply
```

##### B. Invoking CTest

To list all test cases, navigate to the `<build-directory>` or `<build-directory>/runtime`, then do

```
ctest -N runtime-tests
```

To run a specific test case, navigate to `<build-directory>/runtime`, then do

```
ctest -R vec_multiply
```

(picking a test case name from the list printed out by the previously executed `-N` command)

## III. Setup Notes 

1. Clone the repo *with* its submodules

   ```
   git clone --recursive git@github.com:CAPS-UMU/Quidditch.git
   ```

2. ```
   cd quidditch
   ```

3. ```
   mkdir toolchain
   ```

4. ```
   docker run --rm ghcr.io/opencompl/quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
    | tar -xC ./toolchain
   ```

5. ```
   mkdir venv
   virtualenv venv --python=3.11
   ```

6. ```
   source ./venv/bin/activate
   ```

7. ```
   pip install setuptools
   ```

8. ```
   mkdir build && cd build
   ```

9. ```
   cmake .. -GNinja \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
     -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
   ```

10. Build with `-j` set to number of cores (use `nproc --all` to determine number of cores on your system)

   ```
ninja -j 20
   ```

## Old notes below (delete later)

Most recent dev notes up top here.

/home/hoppip/Quidditch/runtime/cmake/quidditch_module.cmake

/home/hoppip/Quidditch/iree/compiler/src/iree/compiler/Dialect/VM/Target/C/CModuleTarget.cpp

/home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Target/LibraryBuilder.h

Line 499: /home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Target/QuidditchTarget.cpp

## Add Test Case caqui

Goal: Let's copy the big_matvec test case, except 

- let's get it to run successfully locally DONE
- let's change the kernels to be ~~the nsnet kernels we want to tile~~ ONE (or two!) nsnet kernel(s) we want to tile
- let's check for correctness
- let's insert cycle counting - DONE

Tried to insert mlir function calls that call C code around a kernel, but I get an IREE error when trying to run my modified vec_multiply:

```
[hoppip@inf-205-141 build]$ ../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/vec_multiply/vec_multiply
[fesvr] Wrote 36 bytes of bootrom to 0x1000
[fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
[fesvr] Wrote 56 bytes of bootdata to 0x1024
[Tracer] Logging Hart          8 to logs/trace_hart_00000008.dasm
[Tracer] Logging Hart          0 to logs/trace_hart_00000000.dasm
[Tracer] Logging Hart          1 to logs/trace_hart_00000001.dasm
[Tracer] Logging Hart          2 to logs/trace_hart_00000002.dasm
[Tracer] Logging Hart          3 to logs/trace_hart_00000003.dasm
[Tracer] Logging Hart          4 to logs/trace_hart_00000004.dasm
[Tracer] Logging Hart          5 to logs/trace_hart_00000005.dasm
[Tracer] Logging Hart          6 to logs/trace_hart_00000006.dasm
[Tracer] Logging Hart          7 to logs/trace_hart_00000007.dasm
iree/runtime/src/iree/vm/context.c:157: NOT_FOUND; required module 'record_cycles' not registered on the context; resolving module 'test_simple_add' imports
[hoppip@inf-205-141 build]$ 
```

