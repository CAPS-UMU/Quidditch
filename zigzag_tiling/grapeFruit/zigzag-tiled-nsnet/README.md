# ZigZag-Tiled NsNet

tile all the nsnet kernels supported by xDSL

## "correct" tiling configs

- ["main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_9_matmul_transpose_b_1x161x600_f64.md)

  ```
  hoodle
  ```

- ["main$async_dispatch_0_matmul_transpose_b_1x400x161_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_0_matmul_transpose_b_1x400x161_f64.md)

  ```
  hoodle
  ```

- ["main$async_dispatch_7_matmul_transpose_b_1x600x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_7_matmul_transpose_b_1x600x400_f64.md)

  ```
  ho0odle
  ```

- ["main$async_dispatch_8_matmul_transpose_b_1x600x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_8_matmul_transpose_b_1x600x600_f64.md)

  ```
  hoodle
  ```

- ["main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_1_matmul_transpose_b_1x1200x400_f64_plan_comparison.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 240;
  l1Tiles[2] = 40;
  l1Interchange = {0, 1, 2}; 
  ```

## old version with incorrect tiling configs

- ["main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_9_matmul_transpose_b_1x161x600_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 161;
  l1Tiles[2] = 100;
  l1Interchange = {0, 1, 2}; 
  ```

  Padding error, so for now use OLD tiling configuration...

- ["main$async_dispatch_0_matmul_transpose_b_1x400x161_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_0_matmul_transpose_b_1x400x161_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 0;
  l1Tiles[2] = 0;
  l1Interchange = {0, 1, 2}; 
  ```

  Apparently this configuration does not fit in L1. I think it is because of the add function following the matmul transpose can I confirm this is why? commenting out for now and using OLD tiling configuration...

- ["main$async_dispatch_7_matmul_transpose_b_1x600x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_7_matmul_transpose_b_1x600x400_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 300;
  l1Tiles[2] = 250;
  l1Interchange = {0, 1, 2}; 
  ```

  Also has padding error. For now, using OLD tiling configuration...

- ["main$async_dispatch_8_matmul_transpose_b_1x600x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_8_matmul_transpose_b_1x600x600_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 30;
  l1Tiles[2] = 200;
  l1Interchange = {0, 1, 2}; 
  ```

  Apparently this configuration does not fit in L1. Error saved below. turning off double buffering and checking if this fixes the problem... After disabling double buffering, I get an mlir lowering error. Disabled ZigZag tiling for now and using OLD tiling configuration...

- ["main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_1_matmul_transpose_b_1x1200x400_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 240;
  l1Tiles[2] = 25;
  l1Interchange = {0, 2, 1}; 
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

## Errors

- ["main$async_dispatch_0_matmul_transpose_b_1x400x161_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_0_matmul_transpose_b_1x400x161_f64.md)

  ```
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

  

- ["main$async_dispatch_8_matmul_transpose_b_1x600x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_8_matmul_transpose_b_1x600x600_f64.md)

  ```
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:105:0: warning: Let's look at ALL the allocOps before doging ANYTHING! 
  
  allocOp with memref shape 1 600 
  
  allocOp with memref shape 1 200 
  
  allocOp with memref shape 1 200 
  
  allocOp with memref shape 30 200 
  
  allocOp with memref shape 30 200 
  
  allocOp with memref shape 1 600 
  
  allocOp with memref shape 1 600 
  Well, those were all the allocOps... =_=
  
  allocOp with memref shape 1 600 
  memref size is 8
  allocElements is 600
  NOW memref size is 4800
  offset is 4800
  
  allocOp with memref shape 1 200 
  memref size is 8
  allocElements is 200
  NOW memref size is 1600
  offset is 6400
  
  allocOp with memref shape 1 200 
  memref size is 8
  allocElements is 200
  NOW memref size is 1600
  offset is 8000
  
  allocOp with memref shape 30 200 
  memref size is 8
  allocElements is 6000
  NOW memref size is 48000
  offset is 56000
  
  allocOp with memref shape 30 200 
  memref size is 8
  allocElements is 6000
  NOW memref size is 48000
  offset is 104000
  
  allocElements is 6000
  memref size is 48000
  offset is 104000
  l1MemoryBytes is 100000, so 4000 too much
  kernel does not fit into L1 memory and cannot be compiled
  /home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
  ```

  After disabling double buffering, I get an mlir lowering error: 

  ```
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:103:0: error: failed to legalize operation 'quidditch_snitch.call_microkernel' that was explicitly marked illegal
  /home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:103:0: note: see current operation: 
  "quidditch_snitch.call_microkernel"(%133, %223, %265) <{
  name = "main$async_dispatch_8_matmul_transpose_b_1x600x600_f64$xdsl_kernel1", 
  riscv_assembly = ".text\0A.globl main$async_dispatch_8_matmul_transpose_b_1x600x600_f64$xdsl_kernel1\0A.p2align 2\0Amain$async_dispatch_8_matmul_transpose_b_1x600x600_f64$xdsl_kernel1:\0A    mv t1, a0\0A    mv t0, a1\0A    li t2, 199\0A    scfgwi t2, 64                                # dm 0 dim 0 bound\0A    li t2, -2\0A    scfgwi t2, 96                                # dm 0 dim 1 bound\0A    li t2, 8\0A    scfgwi t2, 192                               # dm 0 dim 0 stride\0A    li t2, -1592\0A    scfgwi t2, 224                               # dm 0 dim 1 stride\0A    scfgwi zero, 32                              # dm 0 repeat\0A    li t2, -201\0A    scfgwi t2, 65                                # dm 1 dim 0 bound\0A    li t2, 8\0A    scfgwi t2, 193                               # dm 1 dim 0 stride\0A    scfgwi zero, 33                              # dm 1 repeat\0A    scfgwi t1, 800                               # dm 0 dim 1 source\0A    scfgwi t0, 769                               # dm 1 dim 0 source\0A    csrrsi zero, 1984, 1                         # SSR enable\0A    csrrci zero, 1984, 1                         # SSR disable\0A    ret\0A"}> : 
  (memref<1x200xf64>, memref<?x200xf64, strided<[200, 1], offset: ?>>, memref<1x?xf64, strided<[600, 1], offset: ?>>) -> ()
  
  ```

  Context of error in [tail_output2.mlir](../build/tail_output2.mlir)

​	After investigating, problematic pass appears to be `"iree-hal-translate-executables"`

​	check differences between `/home/hoppip/Quidditch/build/before-and-after.mlir` and `/home/hoppip/Quidditch/build/before-and-after-w-error.mlir`([here](build/before-and-after.mlir) and [here](build/before-and-after-w-error.mlir))

--mlir-print-ir-before="quidditch-convert-to-llvm" --mlir-print-ir-after="quidditch-convert-to-llvm"

--mlir-print-ir-before-all

--mlir-print-ir-after-failure

--mlir-print-ir-before="convert-to-llvm" --mlir-print-ir-after="convert-to-llvm"

After using `--mlir-print-ir-after-failure`, it appears that `quidditch-convert-to-llvm` fails before `iree-hal-translate-executables` so problem likely arises from code in this file: `/home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Target/ConvertToLLVM.cpp`

```
RADDISH (q-convert-to-llvm) applyPartialConversion failed :'(
```

see [q-convert-to-llvm-failed.txt](q-convert-to-llvm-failed.txt) for more info

- ["main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_9_matmul_transpose_b_1x161x600_f64.md)

  ```
  <eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:96:0: note: see current operation: %9 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%4, %5 : tensor<1x400xf64>, tensor<600x400xf64>) outs(%8 : tensor<1x600xf64>) -> tensor<1x600xf64>
  // -----// IR Dump After ConcretizePadResultShape Failed (iree-codegen-concretize-pad-result-shape) //----- //
  func.func @main$async_dispatch_9_matmul_transpose_b_1x161x600_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
    %c100 = arith.constant 100 : index
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
      %13 = scf.for %arg2 = %c0 to %c600 step %c100 iter_args(%arg3 = %arg1) -> (tensor<1x168xf64>) {
        %14 = affine.min affine_map<(d0) -> (-d0 + 168, 161)>(%arg0)
        %extracted_slice_2 = tensor.extract_slice %6[0, %arg2] [1, 100] [1, 1] : tensor<1x600xf64> to tensor<1x100xf64>
        %15 = affine.min affine_map<(d0, d1) -> (161, d0 + d1)>(%arg0, %14)
        %16 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%15, %arg0)
        %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%14, %15, %arg0)
        %extracted_slice_3 = tensor.extract_slice %7[%arg0, %arg2] [%16, 100] [1, 1] : tensor<161x600xf64> to tensor<?x100xf64>
        %padded_4 = tensor.pad %extracted_slice_3 low[0, 0] high[%17, 0] {
        ^bb0(%arg4: index, %arg5: index):
          tensor.yield %0 : f64
        } : tensor<?x100xf64> to tensor<?x100xf64>
        %extracted_slice_5 = tensor.extract_slice %arg3[0, %arg0] [1, %14] [1, 1] : tensor<1x168xf64> to tensor<1x?xf64>
        %18 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 161, 100], l1_tiles_interchange = [0, 1, 2], dual_buffer = true>} ins(%extracted_slice_2, %padded_4 : tensor<1x100xf64>, tensor<?x100xf64>) outs(%extracted_slice_5 : tensor<1x?xf64>) -> tensor<1x?xf64>
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

  ## Extra notes
  
  `/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/nsnet2/NsNet2`
  
  vs
  
  `/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit`
  
  NsNet2: cycles 1107880 which is 418306 cycles FASTER than ZigZag tiled nsnet?! Maybe this is because double buffering was turned OFF the last time I ran ZigZag tiled nsnet (grapefruit)?
  
  GrapeFruit: did not finish D:
  
  With double buffering turned off, NsNet takes cycles 1526186, and GrapeFruit takes cycles 1526575, but this is with BOTH tiling the SAME.
  Now I will re-run nsnet with OLD TILING SCHEME to check performance.
  NSNEt2: cycles 1320572
  grapefruit: cycles 1525861
  1525861 - 1320572 = 205289 which is BAD. It means that ZigZag does not recommend optimal tiling here. But this also doesn't account for the addition, either, right?
  
  `1320572 / 1525861= 0.86546 ~ 86.55%`
  
  so NsNet with the OLD tilling scheme is FASTER, but this ZigZag plan is not fully implemented. There is further tiling at L1 that quidditch currently does not do. Can I model Quidditch's tiling with ZigZag, and compare their latency?

