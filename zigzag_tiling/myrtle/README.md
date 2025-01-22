# Myrtle Auto-Tiling

Most recent dev notes up top here.

/home/hoppip/Quidditch/runtime/cmake/quidditch_module.cmake



## Add Test Case caqui

Goal: Let's copy the big_matvec test case, except 

- let's get it to run successfully locally
- let's change the kernels to be the nsnet kernels we want to tile
- let's check for correctness
- let's insert cycle counting



## 0. Daily Use Commands

Inside `Quidditch` directory, do

```
source ./venv/bin/activate
```

After making changes to quidditch, rebuild with 

```
cd build; ninja -j 20
```

 ## I. Run a test case

##### A. Directly Invoking Verilator

Navigate to `<build-directory>` then do

```
../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit
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

