# ZigZag Tiling

This is the landing page for all things ZigZag tiling in Quidditch.

## Daily Use Commands

Inside Quidditch directory, do

```
source ./venv/bin/activate && cd zigzag_tiling
```

After making changes to quidditch, rebuild with 

```
sh aScriptIWillMake.sh
```

Run test case with

```
sh run.sh <nameOfTestToRun>
```

## 0. Setup Notes 

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

10. Build with `-j` set to number of cores
   ```
   ninja -j 20
   ```

   

## I. Manually Tile nsnet with ZigZag

### What are the kernels IREE breaks nsnet into?

TODO

### How does ZigZag recommend these kernels get tiled?

- use [this file](https://github.com/EmilySillars/zigzag/blob/manual-examples/modeling-snitch-with-zigzag.md) as reference, but redo for an L1 cache size of 100,000 bytes and float register files.
- other tiling restrictions: row dimensions must be divisible by 8, make col dimensions as large as possible.

TODO

## II. Automate ZigZag Tiling

TODO

## Troubleshooting 

1. cmake configuration:

   ``` 
   cmake .. -GNinja \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
     -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
   ```

   Invoke build tool:

   ``` 
   cmake --build . -j 20 --target test
   ```

   error:

   ```
   [14/18] Performing build step for 'runtime'
   [0/2] Re-checking globbed directories...
   ninja: error: '/home/hoppip/Quidditch/toolchain/bin/pulp-as', needed by 'samples/big_matvec/big_matvec/big_matvec.o', missing and no known rule to make it
   FAILED: runtime-prefix/src/runtime-stamp/runtime-build /home/hoppip/Quidditch/build/runtime-prefix/src/runtime-stamp/runtime-build 
   cd /home/hoppip/Quidditch/build/runtime && /usr/bin/cmake --build .
   ninja: build stopped: subcommand failed.
   Problem running command: /usr/bin/cmake --build /home/hoppip/Quidditch/build
   Problem executing pre-test command(s).
   Errors while running CTest
   Output from these tests are in: /home/hoppip/Quidditch/build/Testing/Temporary/LastTest.log
   Use "--rerun-failed --output-on-failure" to re-run the failed cases verbosely.
   FAILED: CMakeFiles/test.util 
   cd /home/hoppip/Quidditch/build && /usr/bin/ctest --force-new-ctest-process --verbose
   ninja: build stopped: subcommand failed.
   ```

   See solution to #2.

2. cmake config:

   ```
   cmake .. -GNinja \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
     -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
   ```

   Invoke build tool:
   ```
   ninja -j 20
   ```

   error:
   ```
   -- Build files have been written to: /home/hoppip/Quidditch/build2/runtime
   [14/18] Performing build step for 'runtime'
   [0/2] Re-checking globbed directories...
   ninja: error: '/home/hoppip/Quidditch/toolchain/bin/pulp-as', needed by 'samples/big_matvec/big_matvec/big_matvec.o', missing and no known rule to make it
   FAILED: runtime-prefix/src/runtime-stamp/runtime-build /home/hoppip/Quidditch/build2/runtime-prefix/src/runtime-stamp/runtime-build 
   cd /home/hoppip/Quidditch/build2/runtime && /usr/bin/cmake --build .
   ninja: build stopped: subcommand failed.
   ```

   Investigation:

   ```
   sudo docker pull ghcr.io/opencompl/quidditch/toolchain:main
   
   docker image ls
   
   docker run -it ghcr.io/opencompl/quidditch/toolchain:main
   
   ls /opt/quidditch-toolchain/bin
   ```

   `pulp-as` was indeed inside the docker image!

   But notice I had older docker images as well:
   ```
   [hoppip@inf-205-141 temp]$ sudo docker image ls
   REPOSITORY                              TAG       IMAGE ID       CREATED         SIZE
   ghcr.io/opencompl/quidditch/toolchain   main      7e434df5cfea   8 weeks ago     880MB
   ghcr.io/kuleuven-micas/snax             v0.1.6    fa9b5fdfd4c1   3 months ago    3.74GB
   ghcr.io/opencompl/quidditch/toolchain   <none>    3d6073d33194   5 months ago   
   ```

   Solution:

   ```
   [hoppip@inf-205-141 temp]$ sudo docker rmi -f 7e434df5cfea
   Untagged: ghcr.io/opencompl/quidditch/toolchain:main
   Untagged: ghcr.io/opencompl/quidditch/toolchain@sha256:bd0dc078377136b2cc162210db5e83081a6eeb6ba1f612b651619919f0cd93b9
   Deleted: sha256:7e434df5cfea82bcf0a748a1a8dbb308d8ae1001356cdca58380fe738622e8a6
   [hoppip@inf-205-141 temp]$ sudo docker rmi -f 3d6073d33194
   Untagged: ghcr.io/opencompl/quidditch/toolchain@sha256:f21c91c2c2c92501ac255827167595129d1078f3b40531dbd848427f33f6abe8
   Deleted: sha256:3d6073d3319493c7c757f7d177900a2311f4de2180ee6ff816415985179c246e
   Deleted: sha256:f75e7ec9ab2e4e563f1d127e530a96ce8d8ce801e3cf58728ab5c17578a74e21
   Deleted: sha256:02b420122e1971e21a05072e0041a5deeec17fb3e40e1f397b482b07c6ae22f1
   Deleted: sha256:aedc3bda2944bb9bcb6c3d475bee8b460db9a9b0f3e0b33a6ed2fd1ae0f1d445
   ```

3. ```
   docker run --rm ghcr.io/opencompl/quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
    | tar -xC ./toolchain
   docker: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.
   See 'docker run --help'.
   tar: This does not look like a tar archive
   tar: Exiting with failure status due to previous errors
   ```

   Solution: add `sudo`: 
   ```
   sudo docker run --rm ghcr.io/opencompl/quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
    | tar -xC ./toolchain
   ```

4. ```
   (venv) [hoppip@inf-205-141 Quidditch]$ sudo docker run --rm ghcr.io/opencompl/quidditch/toolchain:main tar -cC /opt/quidditch-toolchain . | tar -xC ./toolchain
   tar: Unexpected EOF in archive
   tar: Unexpected EOF in archive
   tar: Error is not recoverable: exiting now
   ```

   Solution: remove spaces/not printable characters from end of the command. This is a copy and pasting error. Use up arrow to get previous command, then hit delete a couple times, then hit backspace until the `n` in `toolchain` disappears, then retype the `n` and hit enter.

5. ```
   FAILED: snitch_cluster/cluster_gen/snitch_cluster_peripheral.h /home/hoppip/Quidditch/build/runtime/snitch_cluster/cluster_gen/snitch_cluster_peripheral.h 
   cd /home/hoppip/Quidditch/snitch_cluster && /home/hoppip/temp/Quidditch/venv/bin/python3.12 /home/hoppip/Quidditch/snitch_cluster/.bender/git/checkouts/register_interface-4b41cd18fcf60582/vendor/lowrisc_opentitan/util/regtool.py -D -o /home/hoppip/Quidditch/build/runtime/snitch_cluster/cluster_gen/snitch_cluster_peripheral.h /home/hoppip/Quidditch/runtime/../snitch_cluster/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.hjson
   Traceback (most recent call last):
     File "/home/hoppip/Quidditch/snitch_cluster/.bender/git/checkouts/register_interface-4b41cd18fcf60582/vendor/lowrisc_opentitan/util/regtool.py", line 14, in <module>
       from reggen import (
     File "/home/hoppip/Quidditch/snitch_cluster/.bender/git/checkouts/register_interface-4b41cd18fcf60582/vendor/lowrisc_opentitan/util/reggen/gen_dv.py", line 14, in <module>
       from pkg_resources import resource_filename
   ModuleNotFoundError: No module named 'pkg_resources'
   [106/140] Building C object iree-confi...dules_hal_hal.objects.dir/module.c.obj
   ninja: build stopped: subcommand failed.
   FAILED: runtime-prefix/src/runtime-stamp/runtime-build /home/hoppip/Quidditch/build/runtime-prefix/src/runtime-stamp/runtime-build 
   cd /home/hoppip/Quidditch/build/runtime && /usr/bin/cmake --build .
   ninja: build stopped: subcommand failed.
   ```

   Solution:
   ```
   pip install setuptools
   ```

6. ```
   [15/35] Translating NsNet2 using iree-turbine
   FAILED: samples/nsnet2/nsnet2.mlirbc /home/hoppip/Quidditch/build/runtime/samples/nsnet2/nsnet2.mlirbc 
   cd /home/hoppip/Quidditch/build/runtime/samples/nsnet2 && /home/hoppip/temp/Quidditch/venv/bin/python3.12 /home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py /home/hoppip/Quidditch/build/runtime/samples/nsnet2/nsnet2.mlirbc --dtype=f64
   Traceback (most recent call last):
     File "/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py", line 102, in <module>
       exported = aot.export(with_frames(n_frames=args.frames))
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/exporter.py", line 304, in export
       cm = TransformedModule(context=context, import_to="import")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/compiled_module.py", line 652, in __new__
       do_export(proc_def)
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/compiled_module.py", line 649, in do_export
       trace.trace_py_func(invoke_with_self)
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/support/procedural/tracer.py", line 122, in trace_py_func
       return_py_value = _unproxy(py_f(*self.proxy_posargs, **self.proxy_kwargs))
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/compiled_module.py", line 630, in invoke_with_self
       return proc_def.callable(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py", line 91, in main
       y, out1, out2 = aot.jittable(model.forward)(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/support/procedural/base.py", line 135, in __call__
       return current_ir_trace().handle_call(self, args, kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/support/procedural/tracer.py", line 138, in handle_call
       return target.resolve_call(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/shark_turbine/aot/builtins/jittable.py", line 228, in resolve_call
       gm, guards = exported_f(*flat_pytorch_args)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1202, in inner
       check_if_dynamo_supported()
     File "/home/hoppip/temp/Quidditch/venv/lib64/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 593, in check_if_dynamo_supported
       raise RuntimeError("Python 3.12+ not yet supported for torch.compile")
   RuntimeError: Python 3.12+ not yet supported for torch.compile
   [17/35] Generating big_matvec/big_matvec_module.h, big_mat...c.o, big_matvec/big_matvec.h, big_matvec/big_matvec_llvm.h
   ninja: build stopped: subcommand failed.
   FAILED: runtime-prefix/src/runtime-stamp/runtime-build /home/hoppip/Quidditch/build/runtime-prefix/src/runtime-stamp/runtime-build 
   cd /home/hoppip/Quidditch/build/runtime && /usr/bin/cmake --build .
   ninja: build stopped: subcommand failed.
   ```

   Solution: 

   - `deactivate # if inside a venv!`

   - Use [python 3.11](https://www.python.org/downloads/release/python-31110/) instead of 3.12. Install 3.11 if you don't have it.
     While installing python 3.11, I got the error

     ```
     /home/hoppip/Downloads/Python-3.11.10/Modules/_ctypes/_ctypes.c:118:10: fatal error: ffi.h: No such file or directory
       118 | #include <ffi.h>
     ```

     Solution: `sudo dnf install libffi-devel`
     Reality check after installing: `whereis python3.11` also `clear;cat config.log | grep fatal | grep ssl`

   - Use python 3.11 when creating your virtual env:
     ```
     deactivate # if inside a venv!
     cd ..
     rm -r venv; mkdir venv
     virtualenv venv --python=3.11
     ```

     ```
     /usr/lib/python3.11 /usr/lib64/python3.11 /usr/local/bin/python3.11 /usr/local/lib/python3.11
     ```

     ```
     /usr/bin/python3.11 /usr/lib/python3.11 /usr/lib64/python3.11 /usr/local/bin/python3.11 /usr/local/lib/python3.11 /usr/include/python3.11 /usr/share/man/man1/python3.11.1.gz
     
     ```

     ```
     sudo rm -r /usr/lib/python3.11 /usr/lib64/python3.11 /usr/local/bin/python3.11 /usr/local/lib/python3.11
     
     ```

   - Apparently I built `python3.11` without  `openssl` (make sure to check the config.log!! )

     ```
     cmake .. -GNinja   -DCMAKE_C_COMPILER=clang   -DCMAKE_CXX_COMPILER=clang++   -DCMAKE_C_COMPILER_LAUNCHER=ccache   -DCMAKE_CXX_COMPILER_LAUNCHER=ccache   -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
     WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.
     Obtaining file:///home/hoppip/Quidditch/xdsl (from -r /home/hoppip/Quidditch/requirements.txt (line 4))
       Installing build dependencies: started
       Installing build dependencies: finished with status 'error'
       error: subprocess-exited-with-error
       
       × pip subprocess to install build dependencies did not run successfully.
       │ exit code: 1
       ╰─> [12 lines of output]
           WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.
           WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/setuptools/
           WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/setuptools/
           WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/setuptools/
           WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/setuptools/
           WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.")': /simple/setuptools/
           Could not fetch URL https://pypi.org/simple/setuptools/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/setuptools/ (Caused by SSLError("Can't connect to HTTPS URL because the SSL module is not available.")) - skipping
           ERROR: Could not find a version that satisfies the requirement setuptools>=42 (from versions: none)
           ERROR: No matching distribution found for setuptools>=42
     ```

     Solution: Uninstall python3.11, install dependencies, reinstall python3.11
     ```
     sudo dnf remove python3.11
     ```

     Install Dependencies:
     ```
     sudo dnf install git pkg-config
     sudo dnf install dnf-plugins-core
     sudo dnf builddep python3
     ```

     Inside extracted python source folder,
     ```
     ./configure
     clear;cat config.log | grep fatal | grep ssl
     make
     make test
     sudo make install
     ```

7. Warning:

   ```
   [5583/6559] Building CXX object iree-configuration/iree/compiler/plugins/Quidditch/src/Quidditch/Target/CMakeFiles/Quidditch_Target_Passes.objects.dir/PadToTilingConfig.cpp.o
   /home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Target/PadToTilingConfig.cpp:158:19: warning: unused variable '_' [-Wunused-variable]
     158 |     for (unsigned _ : llvm::seq(padOp.getSource().getType().getRank()))
         |     
   ```

   For now, I have ignored this warning.

8. ```
   ninja -j 20
   ```

   Error: 
   ```
   FAILED: iree-configuration/iree/compiler/plugins/Quidditch/src/Quidditch/Dialect/DMA/IR/CMakeFiles/Quidditch_Dialect_DMA_IR_DMADialect.objects.dir/DMAOps.cpp.o 
   /usr/lib64/ccache/c++  -I/home/hoppip/Quidditch/iree -I/home/hoppip/Quidditch/build/codegen/iree-configuration/iree -I/home/hoppip/Quidditch/iree/third_party/llvm-project/llvm/include -I/home/hoppip/Quidditch/build/codegen/iree-configuration/iree/llvm-project/include -I/home/hoppip/Quidditch/iree/third_party/llvm-project/mlir/include -I/home/hoppip/Quidditch/build/codegen/iree-configuration/iree/llvm-project/tools/mlir/include -I/home/hoppip/Quidditch/iree/third_party/llvm-project/lld/include -I/home/hoppip/Quidditch/build/codegen/iree-configuration/iree/llvm-project/tools/lld/include -I/home/hoppip/Quidditch/codegen/compiler/src -I/home/hoppip/Quidditch/build/codegen/iree-configuration/iree/compiler/plugins/Quidditch/src -O3 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden -fno-rtti -fno-exceptions -Wall -Wno-error=deprecated-declarations -Wno-address -Wno-address-of-packed-member -Wno-comment -Wno-format-zero-length -Wno-uninitialized -Wno-overloaded-virtual -Wno-invalid-offsetof -Wno-sign-compare -Wno-unused-function -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-misleading-indentation -fmacro-prefix-map=/home/hoppip/Quidditch/iree=iree -MD -MT iree-configuration/iree/compiler/plugins/Quidditch/src/Quidditch/Dialect/DMA/IR/CMakeFiles/Quidditch_Dialect_DMA_IR_DMADialect.objects.dir/DMAOps.cpp.o -MF iree-configuration/iree/compiler/plugins/Quidditch/src/Quidditch/Dialect/DMA/IR/CMakeFiles/Quidditch_Dialect_DMA_IR_DMADialect.objects.dir/DMAOps.cpp.o.d -o iree-configuration/iree/compiler/plugins/Quidditch/src/Quidditch/Dialect/DMA/IR/CMakeFiles/Quidditch_Dialect_DMA_IR_DMADialect.objects.dir/DMAOps.cpp.o -c /home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Dialect/DMA/IR/DMAOps.cpp
   In file included from /home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Dialect/DMA/IR/DMAOps.cpp:6:
   /home/hoppip/Quidditch/iree/third_party/llvm-project/mlir/include/mlir/Dialect/Linalg/IR/Linalg.h:95:10: fatal error: mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc: No such file or directory
      95 | #include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"
         |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   compilation terminated.
   [289/3498] Building CXX object iree-configuration/iree...KernelOpInterface.objects.dir/UKernelOpInterface.cpp.o
   ninja: build stopped: subcommand failed.
   ```

   Solution: Re-congifure cmake (with clang flags and ccache) and then rebuild:
   ```
   cmake .. -GNinja \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
     -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake
     
   mkdir build && cd build
   
   ninja -j 20
   ```

9. ```
   WARNING: Non-homogeneous multireg PERF_COUNTER_ENABLE skip multireg specific data generation.
   [133/140] Generating nsnet2/nsnet2_module.h, nsnet2/nsnet2...snet2/nsnet2.h, nsnet2/nsnet2_llvm.h, nsnet2/nsnet2_llvm.o
   <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:47:0: warning: Failed to translate kernel with xDSL
   /home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
   <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:47:0: note: see current operation: 
   quidditch_snitch.memref.microkernel(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) : memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>> {
   ^bb0(%arg0: memref<50xf64, strided<[1], offset: ?>>, %arg1: memref<50xf64, strided<[1], offset: ?>>, %arg2: memref<50xf64, strided<[1], offset: ?>>, %arg3: memref<50xf64, strided<[1], offset: ?>>, %arg4: memref<50xf64, strided<[1], offset: ?>>, %arg5: memref<50xf64, strided<[1], offset: ?>>, %arg6: memref<50xf64, strided<[1], offset: ?>>, %arg7: memref<50xf64, strided<[1], offset: ?>>):
     %cst = arith.constant 1.000000e+00 : f64
     linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 : memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>, memref<50xf64, strided<[1], offset: ?>>) outs(%arg7 : memref<50xf64, strided<[1], offset: ?>>) {
     ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
       %0 = arith.addf %in_4, %in_5 : f64
       %1 = arith.addf %in_2, %in_3 : f64
       %2 = arith.negf %1 : f64
       %3 = math.exp %2 : f64
       %4 = arith.addf %3, %cst : f64
       %5 = arith.divf %cst, %4 : f64
       %6 = arith.mulf %in_1, %5 : f64
       %7 = arith.addf %in_0, %6 : f64
       %8 = math.tanh %7 : f64
       %9 = arith.negf %0 : f64
       %10 = math.exp %9 : f64
       %11 = arith.addf %10, %cst : f64
       %12 = arith.divf %cst, %11 : f64
       %13 = arith.subf %in, %8 : f64
       %14 = arith.mulf %13, %12 : f64
       %15 = arith.addf %14, %8 : f64
       linalg.yield %15 : f64
     }
   }
   <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:47:0: note: stderr:
   Traceback (most recent call last):
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/command_line_tool.py", line 534, in parse_chunk
       return self.available_frontends[file_extension](chunk)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/command_line_tool.py", line 520, in parse_mlir
       ).parse_module(not self.args.no_implicit_module)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 127, in parse_module
       if (parsed_op := self.parse_optional_operation()) is not None:
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 675, in parse_optional_operation
       return self.parse_operation()
              ^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 702, in parse_operation
       op = op_type.parse(self)
            ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/dialects/func.py", line 138, in parse
       ) = parse_func_op_like(
           ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/dialects/utils.py", line 239, in parse_func_op_like
       region = parser.parse_optional_region(entry_args)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 539, in parse_optional_region
       self._parse_block_body(entry_block)
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 217, in _parse_block_body
       while (op := self.parse_optional_operation()) is not None:
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 675, in parse_optional_operation
       return self.parse_operation()
              ^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 702, in parse_operation
       op = op_type.parse(self)
            ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/dialects/linalg.py", line 354, in parse
       body = parser.parse_region()
              ^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 594, in parse_region
       region = self.parse_optional_region(arguments)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 558, in parse_optional_region
       block = self._parse_block()
               ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 247, in _parse_block
       self._parse_block_body(block)
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 217, in _parse_block_body
       while (op := self.parse_optional_operation()) is not None:
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 675, in parse_optional_operation
       return self.parse_operation()
              ^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 702, in parse_operation
       op = op_type.parse(self)
            ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/ir/core.py", line 869, in parse
       parser.raise_error(f"Operation {cls.name} does not have a custom format.")
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/base_parser.py", line 107, in raise_error
       raise ParseError(at_position, msg)
   xdsl.utils.exceptions.ParseError: stdin:8:18
       %3 = math.exp %2 : f64
                     ^^
                     Operation math.exp does not have a custom format.
   
   
   The above exception was the direct cause of the following exception:
   
   Traceback (most recent call last):
     File "/home/hoppip/Quidditch/venv/bin/xdsl-opt", line 8, in <module>
       sys.exit(main())
                ^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/xdsl_opt.py", line 5, in main
       xDSLOptMain().run()
     File "/home/hoppip/Quidditch/xdsl/xdsl/xdsl_opt_main.py", line 71, in run
       module = self.parse_chunk(chunk, file_extension, offset)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/command_line_tool.py", line 541, in parse_chunk
       raise Exception("Failed to parse:\n" + e.with_context()) from e
   Exception: Failed to parse:
   stdin:8:18
       %3 = math.exp %2 : f64
                     ^^
                     Operation math.exp does not have a custom format.
   
   
   <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:112:0: warning: Failed to translate kernel with xDSL
   /home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
   <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:112:0: note: see current operation: 
   quidditch_snitch.memref.microkernel(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) : memref<1x21xf64, strided<[168, 1], offset: ?>>, memref<1x21xf64, strided<[168, 1], offset: ?>>, memref<1x21xf64, strided<[168, 1], offset: ?>> {
   ^bb0(%arg0: memref<1x21xf64, strided<[168, 1], offset: ?>>, %arg1: memref<1x21xf64, strided<[168, 1], offset: ?>>, %arg2: memref<1x21xf64, strided<[168, 1], offset: ?>>):
     %cst = arith.constant 1.000000e+00 : f64
     linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x21xf64, strided<[168, 1], offset: ?>>, memref<1x21xf64, strided<[168, 1], offset: ?>>) outs(%arg2 : memref<1x21xf64, strided<[168, 1], offset: ?>>) {
     ^bb0(%in: f64, %in_0: f64, %out: f64):
       %0 = arith.addf %in, %in_0 : f64
       %1 = arith.negf %0 : f64
       %2 = math.exp %1 : f64
       %3 = arith.addf %2, %cst : f64
       %4 = arith.divf %cst, %3 : f64
       linalg.yield %4 : f64
     }
   }
   <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:112:0: note: stderr:
   Traceback (most recent call last):
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/command_line_tool.py", line 534, in parse_chunk
       return self.available_frontends[file_extension](chunk)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/command_line_tool.py", line 520, in parse_mlir
       ).parse_module(not self.args.no_implicit_module)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 127, in parse_module
       if (parsed_op := self.parse_optional_operation()) is not None:
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 675, in parse_optional_operation
       return self.parse_operation()
              ^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 702, in parse_operation
       op = op_type.parse(self)
            ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/dialects/func.py", line 138, in parse
       ) = parse_func_op_like(
           ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/dialects/utils.py", line 239, in parse_func_op_like
       region = parser.parse_optional_region(entry_args)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 539, in parse_optional_region
       self._parse_block_body(entry_block)
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 217, in _parse_block_body
       while (op := self.parse_optional_operation()) is not None:
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 675, in parse_optional_operation
       return self.parse_operation()
              ^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 702, in parse_operation
       op = op_type.parse(self)
            ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/dialects/linalg.py", line 354, in parse
       body = parser.parse_region()
              ^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 594, in parse_region
       region = self.parse_optional_region(arguments)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 558, in parse_optional_region
       block = self._parse_block()
               ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 247, in _parse_block
       self._parse_block_body(block)
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 217, in _parse_block_body
       while (op := self.parse_optional_operation()) is not None:
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 675, in parse_optional_operation
       return self.parse_operation()
              ^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/core.py", line 702, in parse_operation
       op = op_type.parse(self)
            ^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/ir/core.py", line 869, in parse
       parser.raise_error(f"Operation {cls.name} does not have a custom format.")
     File "/home/hoppip/Quidditch/xdsl/xdsl/parser/base_parser.py", line 107, in raise_error
       raise ParseError(at_position, msg)
   xdsl.utils.exceptions.ParseError: stdin:7:18
       %2 = math.exp %1 : f64
                     ^^
                     Operation math.exp does not have a custom format.
   
   
   The above exception was the direct cause of the following exception:
   
   Traceback (most recent call last):
     File "/home/hoppip/Quidditch/venv/bin/xdsl-opt", line 8, in <module>
       sys.exit(main())
                ^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/xdsl_opt.py", line 5, in main
       xDSLOptMain().run()
     File "/home/hoppip/Quidditch/xdsl/xdsl/xdsl_opt_main.py", line 71, in run
       module = self.parse_chunk(chunk, file_extension, offset)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "/home/hoppip/Quidditch/xdsl/xdsl/tools/command_line_tool.py", line 541, in parse_chunk
       raise Exception("Failed to parse:\n" + e.with_context()) from e
   Exception: Failed to parse:
   stdin:7:18
       %2 = math.exp %1 : f64
                     ^^
                     Operation math.exp does not have a custom format.
   
   
   [140/140] Linking C executable samples/nsnet2/NsNet2
   ```

   Who knows.

10. 

