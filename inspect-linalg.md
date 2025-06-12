# Inspecting Linalg

## 0. Hacking on NsNet2

- The NsNet2 test case is located here: [/runtime/samples/nsnet2](/runtime/samples/nsnet2)

- To build NsNet2 *without snitch*, you must make changes to two cmakefiles,

  - `runtime/samples/nsnet2/CMakeLists.txt`
  - `runtime/tests/CMakeLists.txt`
  - All necessary changes documented here: [howToRemoveSnitchTargetFromNsNet2.diff](howToRemoveSnitchTargetFromNsNet2.diff)

- To print out the IR after a specific pass, at the end of the `quidditch_module` rule, add
  ```
  FLAGS -mlir-print-ir-after=convert-elementwise-to-linalg
  ```
  
  and change the "convert-elementwise-to-linalg" to the name of your desired pass.
  
- After changing the CMakeLists.txt file how you like, rebuild from inside your build directory with `ninja`

  - example of build output when printing IR after a pass: [build-output.txt](build-output.txt)

## I. Daily Use Commands

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

 ## II. Run a test case

##### A. Directly Invoking Verilator

Navigate to `<build-directory>` then do

```
../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/nsnet2/NsNet2
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

