# ZigZag-Tiled NsNet

[back to landing page](../../README.md)

We try to tile all the nsnet kernels supported by xDSL, but only manage to tile one kernel (for now)

## regular vs ZigZag-y* tiled nsnet (single kernel)

**Quidditch's `TensorTile` pass does not support multi-level tiling. " Zigzag-y" tile sizes refer to  ZigZag tiling schemes flattened/modified to only tile each dimension once.*

- It turns out only one nsnet kernel can be tiled using zigzag-prescribed tile sizes (otherwise the tiling configuration somehow breaks Quidditch compiler passes farther down in the pipeline).

- We tile this single kernel within NsNet2, `main$async_dispatch_8_matmul_transpose_b_1x600x600_f64`, and compare performance

| Test File Name                                             | Cycle count | [ZigZag Latency Estimate](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_8_matmul_transpose_b_1x600x600_f64_latency_est.md#latency-estimate) |
| ---------------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| NsNet2                                                     | 1110267     | 144731.0                                                     |
| GrapeFruit (NeNet2 with one zigzag-tiled matmul transpose) | 1410139     | 107527.0                                                     |
| NsNet2 / GrapeFruit                                        | 0.787       | 1.346                                                        |

Commands run to get cycle count:

```
/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/nsnet2/NsNet2
/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit
```

## "Zigzag-y" tiling configs

*"ZigZag-y" instead of "ZigZag" because Quidditch's `TensorTile` pass does not support multi-level tiling. The configurations listed below are ZigZag tiling schemes modified to only tile each dimension once.*

- ["main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_9_matmul_transpose_b_1x161x600_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 161;
  l1Tiles[2] = 25;
  l1Interchange = {0, 1, 2};
  ```

  Run into padding error. (More details below)
  Also, ZigZag tiling plan never assigns the output operand to L1. What does that mean? Doesn't it have to be in L1 at some point? What would it mean for an operand to only be assigned at the register file level? When does the output get stored back?

- ["main$async_dispatch_0_matmul_transpose_b_1x400x161_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_0_matmul_transpose_b_1x400x161_f64.md)

  ```
  l1Interchange = {0, 1, 2};
  l1Tiles[0] = 0;
  l1Tiles[1] = 0;
  l1Tiles[2] = 0;
  dualBuffer = false;
  ```

  Doesn't fit in L1 error. (More details below)

- ["main$async_dispatch_7_matmul_transpose_b_1x600x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_7_matmul_transpose_b_1x600x400_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 30;
  l1Tiles[2] = 40;
  l1Interchange = {0, 1, 2}; 
  ```

  Failed to Convert to LLVM error :( (More details below)

- ["main$async_dispatch_8_matmul_transpose_b_1x600x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_8_matmul_transpose_b_1x600x600_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 200;
  l1Tiles[2] = 5;
  l1Interchange = {0, 1, 2}; 
  ```

  No errors :D

- ["main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_1_matmul_transpose_b_1x1200x400_f64_plan_comparison.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 240;
  l1Tiles[2] = 40;
  l1Interchange = {0, 1, 2}; 
  ```

​	Does not fit in L1 error. (More details below)

## Errors

- dispatch 9
  ```
  /home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: %22 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1]>} ins(%17, %18 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  // -----// IR Dump After ConcretizePadResultShape Failed (iree-codegen-concretize-pad-result-shape) //----- //
  func.func @main$async_dispatch_9_matmul_transpose_b_1x161x600_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
    %c25 = arith.constant 25 : index
    %c600 = arith.constant 600 : index
    %c161 = arith.constant 161 : index
    %c168 = arith.constant 168 : index
    %0 = ub.poison : f64
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c4800 = arith.constant 4800 : index
    %c20726400 = arith.constant 20726400 : index
    %c21499200 = arith.constant 21499200 : index
    %c0 = arith.constant 0 : index
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c4800) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x600xf64>>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c20726400) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<161x600xf64>>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c21499200) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x161xf64>>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x161xf64>>
    %5 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [1, 161], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x161xf64>> -> tensor<1x161xf64>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 600], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x600xf64>> -> tensor<1x600xf64>
    %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [161, 600], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<161x600xf64>> -> tensor<161x600xf64>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1, 161], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x161xf64>> -> tensor<1x161xf64>
    %9 = tensor.empty() : tensor<1x168xf64>
    %10 = linalg.fill ins(%cst : f64) outs(%9 : tensor<1x168xf64>) -> tensor<1x168xf64>
    %11 = scf.for %arg0 = %c0 to %c168 step %c161 iter_args(%arg1 = %10) -> (tensor<1x168xf64>) {
      %13 = scf.for %arg2 = %c0 to %c600 step %c25 iter_args(%arg3 = %arg1) -> (tensor<1x168xf64>) {
        %14 = affine.min affine_map<(d0) -> (-d0 + 168, 161)>(%arg0)
        %extracted_slice_2 = tensor.extract_slice %6[0, %arg2] [1, 25] [1, 1] : tensor<1x600xf64> to tensor<1x25xf64>
        %15 = affine.min affine_map<(d0, d1) -> (161, d0 + d1)>(%arg0, %14)
        %16 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%15, %arg0)
        %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%14, %15, %arg0)
        %extracted_slice_3 = tensor.extract_slice %7[%arg0, %arg2] [%16, 25] [1, 1] : tensor<161x600xf64> to tensor<?x25xf64>
        %padded_4 = tensor.pad %extracted_slice_3 low[0, 0] high[%17, 0] {
        ^bb0(%arg4: index, %arg5: index):
          tensor.yield %0 : f64
        } : tensor<?x25xf64> to tensor<?x25xf64>
        %extracted_slice_5 = tensor.extract_slice %arg3[0, %arg0] [1, %14] [1, 1] : tensor<1x168xf64> to tensor<1x?xf64>
        %18 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 161, 25], l1_tiles_interchange = [0, 1, 2], dual_buffer = true>} ins(%extracted_slice_2, %padded_4 : tensor<1x25xf64>, tensor<?x25xf64>) outs(%extracted_slice_5 : tensor<1x?xf64>) -> tensor<1x?xf64>
        %inserted_slice = tensor.insert_slice %18 into %arg3[0, %arg0] [1, %14] [1, 1] : tensor<1x?xf64> into tensor<1x168xf64>
        scf.yield %inserted_slice : tensor<1x168xf64>
      }
      scf.yield %13 : tensor<1x168xf64>
    }
    %padded = tensor.pad %8 low[0, 0] high[0, 7] {
    ^bb0(%arg0: index, %arg1: index):
      tensor.yield %0 : f64
    } : tensor<1x161xf64> to tensor<1x168xf64>
    %padded_1 = tensor.pad %5 low[0, 0] high[0, 7] {
    ^bb0(%arg0: index, %arg1: index):
      tensor.yield %0 : f64
    } : tensor<1x161xf64> to tensor<1x168xf64>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %padded : tensor<1x168xf64>, tensor<1x168xf64>) outs(%padded_1 : tensor<1x168xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %13 = arith.addf %in, %in_2 : f64
      %14 = arith.negf %13 : f64
      %15 = math.exp %14 : f64
      %16 = arith.addf %15, %cst_0 : f64
      %17 = arith.divf %cst_0, %16 : f64
      linalg.yield %17 : f64
    } -> tensor<1x168xf64>
    %extracted_slice = tensor.extract_slice %12[0, 0] [1, 161] [1, 1] : tensor<1x168xf64> to tensor<1x161xf64>
    flow.dispatch.tensor.store %extracted_slice, %4, offsets = [0, 0], sizes = [1, 161], strides = [1, 1] : tensor<1x161xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x161xf64>>
    return
  }
  ```

- dispatch 0
  ```
  /home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:10:0: note: see current operation: %9 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40], l1_tiles_interchange = [0, 1]>} ins(%4, %5 : tensor<1x161xf64>, tensor<400x161xf64>) outs(%8 : tensor<1x400xf64>) -> tensor<1x400xf64>
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:10:0: warning: Let's look at ALL the allocOps before doging ANYTHING! 
  
  allocOp with memref shape 1 400 
  
  allocOp with memref shape 1 161 
  
  allocOp with memref shape 400 161 
  
  allocOp with memref shape 1 400 
  
  allocOp with memref shape 1 400 
  Well, those were all the allocOps... =_=
  
  allocOp with memref shape 1 400 
  memref size is 8
  allocElements is 400
  NOW memref size is 3200
  offset is 3200
  
  allocOp with memref shape 1 161 
  memref size is 8
  allocElements is 161
  NOW memref size is 1288
  offset is 4488
  
  allocOp with memref shape 400 161 
  memref size is 8
  allocElements is 64400
  NOW memref size is 515200
  offset is 519744
  
  allocElements is 64400
  memref size is 515200
  offset is 519744
  l1MemoryBytes is 100000, so 419744 too much
  kernel does not fit into L1 memory and cannot be compiled
  ```

  

- dispatch 7
  ```
  FAILED: samples/grapeFruit/grapeFruit/grapeFruit_module.h samples/grapeFruit/grapeFruit/grapeFruit.o samples/grapeFruit/grapeFruit/grapeFruit.h samples/grapeFruit/grapeFruit/grapeFruit_llvm.h samples/grapeFruit/grapeFruit/grapeFruit_llvm.o /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit_module.h /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit.o /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit.h /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit_llvm.h /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit_llvm.o 
  cd /home/hoppip/Quidditch/build/runtime/samples/grapeFruit && /home/hoppip/Quidditch/build/codegen/iree-configuration/iree/tools/iree-compile --mlir-print-ir-after-failure --iree-quidditch-zigzag-tiling-schemes=/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/zigzag-tiled-nsnet/zigzag-tiled-nsnet.json --iree-vm-bytecode-module-strip-source-map=true --iree-vm-emit-polyglot-zip=false --iree-input-type=auto --iree-input-demote-f64-to-f32=0 --iree-hal-target-backends=quidditch --iree-quidditch-static-library-output-path=/home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit.o --iree-quidditch-xdsl-opt-path=/home/hoppip/Quidditch/venv/bin/xdsl-opt --iree-quidditch-toolchain-root=/home/hoppip/Quidditch/toolchain --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-debug-symbols=true --iree-llvmcpu-target-triple=riscv32-unknown-elf --iree-llvmcpu-target-cpu=generic-rv32 --iree-llvmcpu-target-cpu-features=+m,+f,+d,+zfh --iree-llvmcpu-target-abi=ilp32d --iree-llvmcpu-target-float-abi=hard --iree-llvmcpu-link-embedded=false --iree-llvmcpu-link-static --iree-llvmcpu-number-of-threads=8 --iree-llvmcpu-static-library-output-path=/home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit_llvm.o --output-format=vm-c --iree-vm-target-index-bits=32 /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit.mlirbc -o /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/grapeFruit/grapeFruit_module.h
  
  // -----// IR Dump After ConvertToLLVMPass Failed (quidditch-convert-to-llvm) //----- //
  module attributes {llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-elf"} {
    func.func @main$async_dispatch_7_matmul_transpose_b_1x600x400_f64() attributes {quidditch_snitch.dma_specialization = @main$async_dispatch_7_matmul_transpose_b_1x600x400_f64$dma, translation_info = #iree_codegen.translation_info<None>} {
  ```
  
  
  
- dispatch 8

- dispatch 1

  ```
  [37/41] Generating grapeFruit/grapeFruit_module.h, grapeFruit/grapeFruit.o, grapeFruit/grapeFruit.h, grapeFruit/grapeFruit_llvm.h, grapeFruit/grapeFruit_llvm.o
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Let's look at ALL the allocOps before doging ANYTHING! 
  
  allocOp with memref shape 1 1200 
  
  allocOp with memref shape 1 40 
  
  allocOp with memref shape 240 40 
  
  allocOp with memref shape 1 1200 
  
  allocOp with memref shape 1 1200 
  Well, those were all the allocOps... =_=
  
  allocOp with memref shape 1 1200 
  memref size is 8
  allocElements is 1200
  NOW memref size is 9600
  offset is 9600
  
  allocOp with memref shape 1 40 
  memref size is 8
  allocElements is 40
  NOW memref size is 320
  offset is 9920
  
  allocOp with memref shape 240 40 
  memref size is 8
  allocElements is 9600
  NOW memref size is 76800
  offset is 86720
  
  allocOp with memref shape 1 1200 
  memref size is 8
  allocElements is 1200
  NOW memref size is 9600
  offset is 96320
  
  allocOp with memref shape 1 1200 
  memref size is 8
  allocElements is 1200
  NOW memref size is 9600
  offset is 105920
  
  allocElements is 1200
  memref size is 9600
  offset is 105920
  l1MemoryBytes is 100000, so 5920 too much
  kernel does not fit into L1 memory and cannot be compiled
  /home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
  "func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
  ```

  

## ZigZag-Tiled vs Quidditch-Tiled for one kernel

```
/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/nsnet2/NsNet2
/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit
```

With double buffering turned off, 
NsNet2: cycles 1320572
GrapeFruit: cycles 1525861
1525861 - 1320572 = 205289 which is BAD. It means that ZigZag supposedly does not recommend the optimal tiling here. 

"Speedup" for this case is `1320572 / 1525861= 0.86546 ~ 86.55%`

So NsNet2 with the OLD tilling scheme is FASTER, but also the ZigZag plan given to GrapeFruit is not fully implemented. There is further tiling at L1 that Quidditch currently does not do. Can I model Quidditch's tiling with ZigZag, and compare that estimated latency with Grapefruit's estimated latency?
