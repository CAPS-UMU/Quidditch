# ZigZag-Tiled NsNet

tile all the nsnet kernels supported by xDSL

- ["main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_9_matmul_transpose_b_1x161x600_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 161;
  l1Tiles[2] = 100;
  l1Interchange = {0, 1, 2}; 
  ```

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

- ["main$async_dispatch_8_matmul_transpose_b_1x600x600_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_8_matmul_transpose_b_1x600x600_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 30;
  l1Tiles[2] = 200;
  l1Interchange = {0, 1, 2}; 
  ```

  Apparently this configuration does not fit in L1. Error saved below. turning off double buffering and checking if this fixes the problem... After disabling double buffering, I get an mlir lowering error. Disabled ZigZag tiling for now.

- ["main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"](https://github.com/EmilySillars/zigzag/blob/manual-examples/tiling-nsnet/dispatch_1_matmul_transpose_b_1x1200x400_f64.md)

  ```
  l1Tiles[0] = 0;
  l1Tiles[1] = 240;
  l1Tiles[2] = 25;
  l1Interchange = {0, 2, 1}; 
  ```

  

## ZigZag-Tiled vs Quidditch-Tiled for (as many kernels as we can)



`/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/nsnet2/NsNet2`

vs

`/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit`



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



