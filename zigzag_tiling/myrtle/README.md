# Myrtle Auto-Tiling



## How to Test Myrtle with Different Tile Sizes

1. Navigate to the comparing-tile-sizes directory
   ```
   cd comparing-tile-sizes
   ```

2. Add tiling schemes (as json files) to the folder `tile-sizes-to-test`

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

