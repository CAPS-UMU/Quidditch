# grapefruit notes

Questions I have:

1. what does the IR look like for "fits-right" AFTER the `LowerL1Allocations` pass without double buffering? what does the IR look like AFTER the removal of redundant L1 requests? Which pass refers to this?
2. what does "too-big" look like without double buffering AND without the add function?

## main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64
As ZigZag workload:
```
- id: 0 
  name: dispatch_1_matmul_transpose_b_1x1200x400_f64  # name can be used to specify mapping
  operator_type: default  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[b][c]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [1, 1200, 400]
  operand_precision:
    W: 64
    I: 64
    O: 64
    O_final: 64
  operand_source:
    I: 0
    W: 0
```
What is the hardware description used? Default temporal and spatial mappings?

- ZigZag run documented [here](https://github.com/EmilySillars/zigzag/blob/manual-examples/modeling-snitch-with-zigzag.md#dispatch_1_matmul_transpose_b_1x1200x400_f64).

- Full ZigZag output as a json file [here](snitch-cluster-only-floats-no-ssrs-dispatch_1_matmul_transpose_b_1x1200x400_f64/dispatch_1_matmul_transpose_b_1x1200x400_f64_complete.json)

```
hoodleLoop ordering for dispatch_1_matmul_transpose_b_1x1200x400_f64
=============================================================================================
Temporal Loops                      O                  W                  I                  
=============================================================================================
for C in [0, 5):                    l1                 l3                 l1                 
---------------------------------------------------------------------------------------------
  for B in [0, 5):                  l1                 l3                 l1                 
---------------------------------------------------------------------------------------------
    for C in [0, 5):                rf_f0_thru_f31     l1                 l1                 
---------------------------------------------------------------------------------------------
      for C in [0, 16):             rf_f0_thru_f31     l1                 l1                 
---------------------------------------------------------------------------------------------
        for B in [0, 6):            rf_f0_thru_f31     l1                 rf_f0_thru_f31     
---------------------------------------------------------------------------------------------
          for B in [0, 5):          rf_f0_thru_f31     l1                 rf_f0_thru_f31     
---------------------------------------------------------------------------------------------
=============================================================================================
Spatial Loops                                                                                
=============================================================================================
            parfor B in [0, 8):                                                              
---------------------------------------------------------------------------------------------
            parfor C in [0, 1):                                                              
---------------------------------------------------------------------------------------------
```

Sanity Check: Does the tiling at least (sort of) make sense?

```
Bounds for A: none; dimension A has a cardinality of 1, so not tiling this dimension checks out OK.

Bounds for B: [5, 6, 5, 8 (spatial)]; 5*6*5*8 = 1200, so this checks out OK.

Bounds for C: [5, 5, 16, 1 (degenerate)]; 5*5*16 = 400, so this checks out OK.
```

What are the tile sizes for each dimension then?

```
Tile Sizes for A: [1] (or 0, if input to the upstream mlir tiling function)
Tile Sizes for B: [240, 40, 8, 1 (spatial)]
Tile Sizes for C: [80, 16, 1 (spatial)]
```

What are the bounds and tile sizes, taking into account that Quidditch only cares about tiling L3 to fit into L1?

```
Bounds for A: []
Bounds for B: [5]
Bounds for C: [5]
Tile Sizes for A: [1] (or 0, if input to the upstream mlir tiling function)
Tile Sizes for B: [240]
Tile Sizes for C: [80]
```

Let's summarize this as an input json:

```
{
    "bounds":[[], [5], [5]],
    "order":[[0,0], [2,0], [1,0]]
}
```

Note: 0 refers to original A loop, 1 to original B loop, and 2 to original C loop.

## manually tile nsnet kernel with ZigZag pass

Operation before:

```
%23 = linalg.matmul_transpose_b {
lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], 

l1_tiles_interchange = [2, 0, 1], 
dual_buffer = true>} 
ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) 
outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
```

To be explicit, this means that

```
Input matrix has size 1x400

Weight matrix has size 1200x400

Output matrix has size 1x1200
```

Operation after:

```
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %c0 = arith.constant 0 : index
  %c400 = arith.constant 400 : index
  %c80 = arith.constant 80 : index
  
  %23 = scf.for %arg0 = %c0 to %c400 step %c80 iter_args(%arg1 = %22) -> (tensor<1x1200xf64>) {
    %c0_0 = arith.constant 0 : index
    %c1200 = arith.constant 1200 : index
    %c240 = arith.constant 240 : index
    
    %25 = scf.for %arg2 = %c0_0 to %c1200 step %c240 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
      %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, 80] [1, 1] 
      					: tensor<1x400xf64> to tensor<1x80xf64>
      %extracted_slice_1 = tensor.extract_slice %19[%arg2, %arg0] [240, 80] [1, 1] 
      					: tensor<1200x400xf64> to tensor<240x80xf64>
      %extracted_slice_2 = tensor.extract_slice %arg3[0, %arg2] [1, 240] [1, 1] 
      					: tensor<1x1200xf64> to tensor<1x240xf64>
      					
      %26 = linalg.matmul_transpose_b 
      		ins(%extracted_slice, %extracted_slice_1 : tensor<1x80xf64>, tensor<240x80xf64>) 
      		outs(%extracted_slice_2 : tensor<1x240xf64>) -> tensor<1x240xf64>
      		
      %inserted_slice = tensor.insert_slice %26 into %arg3[0, %arg2] [1, 240] [1, 1] 
      					: tensor<1x240xf64> into tensor<1x1200xf64>
      scf.yield %inserted_slice : tensor<1x1200xf64>
    }
    scf.yield %25 : tensor<1x1200xf64>
  }
```

Running grapefruit with zigzag-tiled kernel:

```
/home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit
```

If the program finishes (without errors), compare two versions.

Else,  try to re-do using modified `configureForSnitch.cpp`. 

​	If program still fails/and/or/hangs, try Shreya pytorch method

​	Else, compare two versions.

## What if I modify configureForSnitch to use zigzag tiling config?

- I get this error: kernel does not fit into L1 memory and cannot be compiled
- [Full error here](error-after-modifying-config-for-snitch.mlir)
- From: /home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Dialect/Snitch/Transforms/LowerL1Allocations.cpp

```
  %32 = "dma.start_transfer"(%22, %31) : (
  memref<1x1200xf64, strided<[1200, 1], offset: ?>>, 
  memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
 
```



Next step: print out more info. For example, what is the memref we try to allocate that means there isn't enough space? Use string stream

```
allocOp with memref shape 1 1200 
offset is 0

allocOp with memref shape 1 80 
offset is 9600

allocOp with memref shape 240 80 
offset is 10240

allocElements is 19200
memref size is 153600
offset is 163840
l1MemoryBytes is 100000, so 63840too much
kernel does not fit into L1 memory and cannot be compiled
```

NEXT STEP:

**Why is it trying to allocate 1x1200 in L1? I am pretty sure there only needs to be tensor<1x240xf64>. Can we print out ALL the allocops with their sizes FIRST, then try to allocate, so we can check which allocop statements there actually are? I don't see the 1x400 in here. Is that because it appears later in the list, or because there is no allocaop for the 1x400?**

## I messed up and gave zigzag the wrong workload

ZigZag run documented [here](https://github.com/EmilySillars/zigzag/blob/manual-examples/modeling-snitch-with-zigzag.md#dispatch_1_matmul_transpose_b_1x1200x400_f64).

What is the output from ZigZag, as ASCII and in the JSON file?
```
Loop ordering for dispatch_1_matmul_transpose_b_1x1200x400_f64
=============================================================================================
Temporal Loops                      O                  W                  I                  
=============================================================================================
for C in [0, 5):                    l1                 l3                 l1                 
---------------------------------------------------------------------------------------------
  for B in [0, 16):                 l1                 l3                 l1                 
---------------------------------------------------------------------------------------------
    for C in [0, 5):                rf_f0_thru_f31     l1                 l1                 
---------------------------------------------------------------------------------------------
      for C in [0, 6):              rf_f0_thru_f31     l1                 rf_f0_thru_f31     
---------------------------------------------------------------------------------------------
        for B in [0, 5):            rf_f0_thru_f31     l1                 rf_f0_thru_f31     
---------------------------------------------------------------------------------------------
          for B in [0, 5):          rf_f0_thru_f31     l1                 rf_f0_thru_f31     
---------------------------------------------------------------------------------------------
=============================================================================================
Spatial Loops                                                                                
=============================================================================================
            parfor C in [0, 8):                                                              
---------------------------------------------------------------------------------------------
            parfor A in [0, 1):                                                              
---------------------------------------------------------------------------------------------

```
Full json file [here](snitch-cluster-only-floats-no-ssrs-dispatch_1_matmul_transpose_b_1x1200x400_f64/dispatch_1_matmul_transpose_b_1x1200x400_f64_complete.json).

Note that we only really care about the tiling from ZigZag until the operands reach L1 + spatial unrolling.
```
Loop ordering for dispatch_1_matmul_transpose_b_1x1200x400_f64
=============================================================================================
Temporal Loops                      O                  W                  I                  
=============================================================================================
for C in [0, 5):                    l1                 l3                 l1                 
---------------------------------------------------------------------------------------------
  for B in [0, 16):                 l1                 l3                 l1                 
---------------------------------------------------------------------------------------------
    for C in [0, 5):                rf_f0_thru_f31     l1                 l1                 
---------------------------------------------------------------------------------------------

=============================================================================================
Spatial Loops                                                                                
=============================================================================================
            parfor C in [0, 8):                                                              
---------------------------------------------------------------------------------------------
            parfor A in [0, 1):                                                              
---------------------------------------------------------------------------------------------
```

In my simplified json format, what would this tiling scheme look like?
```
{
    "bounds":[[1], [16], [5]],
    "order":[[0,0], [2,0], [1,0], [1,1]]
}
```

which means the tile sizes are 1/1 = 1, 400/16 = 25, 1200 / 5 = 240, or as input to mlir tiling func: [0, 25, 240]

and I THINK interchange would be 0, 2, 1, but we will make sure. 0 apparently corresponds to the outermost loop.  leave A along, then then place loops in order C followed by B, which is 0, 2, 1.

The BEFORE operation:

```
  %23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
```

The AFTER operation:

```
  %c0 = arith.constant 0 : index
  %c400 = arith.constant 400 : index
  %c240 = arith.constant 240 : index
  %23 = scf.for %arg0 = %c0 to %c400 step %c240 iter_args(%arg1 = %22) -> (tensor<1x1200xf64>) {
    %c0_0 = arith.constant 0 : index
    %c1200 = arith.constant 1200 : index
    %c25 = arith.constant 25 : index
    %25 = scf.for %arg2 = %c0_0 to %c1200 step %c25 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
      %c400_1 = arith.constant 400 : index
      %26 = affine.min affine_map<(d0) -> (240, -d0 + 400)>(%arg0)
      %27 = affine.apply affine_map<(d0) -> (d0 - 1)>(%26)
      %28 = affine.apply affine_map<(d0) -> (d0 - 1)>(%26)
      %29 = affine.apply affine_map<(d0) -> (d0 - 1)>(%26)
      %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, %26] [1, 1] : tensor<1x400xf64> to tensor<1x?xf64>
      %extracted_slice_2 = tensor.extract_slice %19[%arg2, %arg0] [25, %26] [1, 1] : tensor<1200x400xf64> to tensor<25x?xf64>
      %extracted_slice_3 = tensor.extract_slice %arg3[0, %arg2] [1, 25] [1, 1] : tensor<1x1200xf64> to tensor<1x25xf64>
      %30 = linalg.matmul_transpose_b ins(%extracted_slice, %extracted_slice_2 : tensor<1x?xf64>, tensor<25x?xf64>) outs(%extracted_slice_3 : tensor<1x25xf64>) -> tensor<1x25xf64>
      %31 = affine.apply affine_map<(d0) -> (d0 - 1)>(%26)
      %inserted_slice = tensor.insert_slice %30 into %arg3[0, %arg2] [1, 25] [1, 1] : tensor<1x25xf64> into tensor<1x1200xf64>
      scf.yield %inserted_slice : tensor<1x1200xf64>
    }
    scf.yield %25 : tensor<1x1200xf64>
  }
```

Did I perform the tiling correctly? Obviously there are dynamic tensor sizes (which are NOT okay), but are there any other problems?

## Transform kernel based on ZigZag tiling scheme

Isolated kernel with this warning here:

```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: %23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
```

Transformed it into:

```
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
%24 = scf.for %arg0 = %c0 to %c400 step %c240 iter_args(%arg1 = %22) -> (tensor<1x1200xf64>) {
  %c0_0 = arith.constant 0 : index
  %c1200 = arith.constant 1200 : index
  %c25 = arith.constant 25 : index
  %26 = scf.for %arg2 = %c0_0 to %c1200 step %c25 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
    %c400_1 = arith.constant 400 : index
    %27 = affine.min affine_map<(d0) -> (240, -d0 + 400)>(%arg0)
    %28 = affine.apply affine_map<(d0) -> (d0 - 1)>(%27)
    %29 = affine.apply affine_map<(d0) -> (d0 - 1)>(%27)
    %30 = affine.apply affine_map<(d0) -> (d0 - 1)>(%27)
    %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, %27] [1, 1] : tensor<1x400xf64> to tensor<1x?xf64>
    %extracted_slice_2 = tensor.extract_slice %19[%arg2, %arg0] [25, %27] [1, 1] : tensor<1200x400xf64> to tensor<25x?xf64>
    %extracted_slice_3 = tensor.extract_slice %arg3[0, %arg2] [1, 25] [1, 1] : tensor<1x1200xf64> to tensor<1x25xf64>
    %31 = linalg.matmul_transpose_b ins(%extracted_slice, %extracted_slice_2 : tensor<1x?xf64>, tensor<25x?xf64>) outs(%extracted_slice_3 : tensor<1x25xf64>) -> tensor<1x25xf64>
    %32 = affine.apply affine_map<(d0) -> (d0 - 1)>(%27)
    %inserted_slice = tensor.insert_slice %31 into %arg3[0, %arg2] [1, 25] [1, 1] : tensor<1x25xf64> into tensor<1x1200xf64>
    scf.yield %inserted_slice : tensor<1x1200xf64>
  }
  scf.yield %26 : tensor<1x1200xf64>
}
```

Which throws the error:

```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: error: 'memref.alloca' op L1 allocations with dynamic size is currently unsupported
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 

%40 = "memref.alloca"(%38) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<1x?xf64, #quidditch_snitch.l1_encoding>

<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: error: 'memref.alloca' op L1 allocations with dynamic size is currently unsupported
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 

%45 = "memref.alloca"(%38) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<25x?xf64, #quidditch_snitch.l1_encoding>

<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: error: failed to run translation of source executable to target executable for backend #hal.executable.target<"quidditch", "static", {compute_cores = 8 : i32, data_layout = "e-m:e-p:32:32-i64:64-n32-S128", target_triple = "riscv32-unknown-elf"}>
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
"hal.executable.variant"() ({
  "hal.executable.export"() ({
  ^bb0(%arg20: !hal.device):
    %66 = "arith.constant"() <{value = 1 : index}> : () -> index
    "hal.return"(%66, %66, %66) : (index, index, index) -> ()
  }) {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], layout = #hal.pipeline.layout<push_constants = 5, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>, ordinal = 0 : index, sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"} : () -> ()
  "builtin.module"() ({
    "func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
      %0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>
      %1 = "arith.constant"() <{value = 32 : index}> : () -> index
      %2 = "arith.constant"() <{value = 25 : index}> : () -> index
      %3 = "arith.constant"() <{value = 1200 : index}> : () -> index
      %4 = "arith.constant"() <{value = 240 : index}> : () -> index
      %5 = "arith.constant"() <{value = 400 : index}> : () -> index
      %6 = "arith.constant"() <{value = 0 : index}> : () -> index
      %7 = "arith.constant"() <{value = 32 : i64}> : () -> i64
      %8 = "hal.interface.constant.load"() {index = 0 : index} : () -> i32
      %9 = "hal.interface.constant.load"() {index = 1 : index} : () -> i32
      %10 = "hal.interface.constant.load"() {index = 2 : index} : () -> i32
      %11 = "hal.interface.constant.load"() {index = 3 : index} : () -> i32
      %12 = "hal.interface.constant.load"() {index = 4 : index} : () -> i32
      %13 = "arith.extui"(%8) : (i32) -> i64
      %14 = "arith.extui"(%9) : (i32) -> i64
      %15 = "arith.shli"(%14, %7) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %16 = "arith.ori"(%13, %15) : (i64, i64) -> i64
      %17 = "arith.index_castui"(%16) {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : (i64) -> index
      %18 = "arith.index_castui"(%10) : (i32) -> index
      %19 = "arith.index_castui"(%11) : (i32) -> index
      %20 = "arith.index_castui"(%12) : (i32) -> index
      %21 = "hal.interface.binding.subspan"(%17) {alignment = 64 : index, binding = 0 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x400xf64, strided<[400, 1], offset: ?>>
      "memref.assume_alignment"(%21) <{alignment = 64 : i32}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>) -> ()
      %22 = "hal.interface.binding.subspan"(%18) {alignment = 64 : index, binding = 1 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1200x400xf64, strided<[400, 1], offset: ?>>
      "memref.assume_alignment"(%22) <{alignment = 1 : i32}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>) -> ()
      %23 = "hal.interface.binding.subspan"(%19) {alignment = 64 : index, binding = 1 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
      "memref.assume_alignment"(%23) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
      %24 = "hal.interface.binding.subspan"(%20) {alignment = 64 : index, binding = 2 : index, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
      "memref.assume_alignment"(%24) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
      %25 = "arith.constant"() <{value = 0 : index}> : () -> index
      %26 = "memref.view"(%0, %25) : (memref<100000xi8>, index) -> memref<1200xf64>
      %27 = "memref.reinterpret_cast"(%26) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
      %28 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64>
      %29 = "quidditch_snitch.compute_core_index"() : () -> index
      %30 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 150)>}> : (index) -> index
      "scf.for"(%30, %3, %3) ({
      ^bb0(%arg16: index):
        %64 = "memref.subview"(%27, %arg16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        "quidditch_snitch.memref.microkernel"(%64) ({
        ^bb0(%arg17: memref<1x150xf64, strided<[1200, 1], offset: ?>>):
          %65 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
          "linalg.fill"(%65, %arg17) <{operandSegmentSizes = array<i32: 1, 1>}> ({
          ^bb0(%arg18: f64, %arg19: f64):
            "linalg.yield"(%arg18) : (f64) -> ()
          }) : (f64, memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%6, %5, %4) ({
      ^bb0(%arg7: index):
        %48 = "affine.min"(%arg7) <{map = affine_map<(d0) -> (-d0 + 400, 240)>}> : (index) -> index
        %49 = "memref.subview"(%21, %arg7, %48) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<1x?xf64, strided<[400, 1], offset: ?>>
        %50 = "memref.alloca"(%48) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<1x?xf64>
        %51 = "memref.subview"(%50, %48) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x?xf64>, index) -> memref<1x?xf64, strided<[?, 1]>>
        %52 = "dma.start_transfer"(%49, %51) : (memref<1x?xf64, strided<[400, 1], offset: ?>>, memref<1x?xf64, strided<[?, 1]>>) -> !dma.token
        "dma.wait_for_transfer"(%52) : (!dma.token) -> ()
        "scf.for"(%6, %3, %2) ({
        ^bb0(%arg8: index):
          %53 = "memref.subview"(%22, %arg8, %arg7, %48) <{operandSegmentSizes = array<i32: 1, 2, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index, index) -> memref<25x?xf64, strided<[400, 1], offset: ?>>
          %54 = "memref.subview"(%27, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 25>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x25xf64, strided<[1200, 1], offset: ?>>
          %55 = "memref.alloca"(%48) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<25x?xf64>
          %56 = "memref.subview"(%55, %48) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<25x?xf64>, index) -> memref<25x?xf64, strided<[?, 1]>>
          %57 = "dma.start_transfer"(%53, %56) : (memref<25x?xf64, strided<[400, 1], offset: ?>>, memref<25x?xf64, strided<[?, 1]>>) -> !dma.token
          "dma.wait_for_transfer"(%57) : (!dma.token) -> ()
          %58 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 4)>}> : (index) -> index
          "scf.for"(%58, %2, %1) ({
          ^bb0(%arg9: index):
            %59 = "affine.min"(%arg9) <{map = affine_map<(d0) -> (-d0 + 25, 4)>}> : (index) -> index
            %60 = "memref.subview"(%55, %arg9, %59, %48) <{operandSegmentSizes = array<i32: 1, 1, 2, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<25x?xf64>, index, index, index) -> memref<?x?xf64, strided<[?, 1], offset: ?>>
            %61 = "memref.subview"(%54, %arg9, %59) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x25xf64, strided<[1200, 1], offset: ?>>, index, index) -> memref<1x?xf64, strided<[1200, 1], offset: ?>>
            "quidditch_snitch.memref.microkernel"(%51, %60, %61) ({
            ^bb0(%arg10: memref<1x?xf64, strided<[?, 1]>>, %arg11: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg12: memref<1x?xf64, strided<[1200, 1], offset: ?>>):
              "linalg.matmul_transpose_b"(%arg10, %arg11, %arg12) <{operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
                %62 = "arith.mulf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
                %63 = "arith.addf"(%arg15, %62) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
                "linalg.yield"(%63) : (f64) -> ()
              }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (memref<1x?xf64, strided<[?, 1]>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<1x?xf64, strided<[1200, 1], offset: ?>>) -> ()
            }) : (memref<1x?xf64, strided<[?, 1]>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<1x?xf64, strided<[1200, 1], offset: ?>>) -> ()
            "quidditch_snitch.microkernel_fence"() : () -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %31 = "arith.constant"() <{value = 9600 : index}> : () -> index
      %32 = "memref.view"(%0, %31) : (memref<100000xi8>, index) -> memref<1200xf64>
      %33 = "memref.reinterpret_cast"(%32) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
      %34 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64>
      %35 = "memref.cast"(%33) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>>
      %36 = "dma.start_transfer"(%23, %35) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>>) -> !dma.token
      "dma.wait_for_transfer"(%36) : (!dma.token) -> ()
      %37 = "arith.constant"() <{value = 19200 : index}> : () -> index
      %38 = "memref.view"(%0, %37) : (memref<100000xi8>, index) -> memref<1200xf64>
      %39 = "memref.reinterpret_cast"(%38) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
      
      %40 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64>
     
     %41 = "memref.cast"(%39) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>>
      %42 = "dma.start_transfer"(%24, %41) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>>) -> !dma.token
      "dma.wait_for_transfer"(%42) : (!dma.token) -> ()
      "scf.for"(%30, %3, %3) ({
      ^bb0(%arg0: index):
        %44 = "memref.subview"(%27, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        
        %45 = "memref.subview"(%33, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
       
       %46 = "memref.subview"(%39, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        "quidditch_snitch.memref.microkernel"(%44, %45, %46) ({
        ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>>):
          "linalg.generic"(%arg1, %arg2, %arg3) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):
            %47 = "arith.addf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%47) : (f64) -> ()
          }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %43 = "dma.start_transfer"(%39, %24) : (memref<1x1200xf64>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
      "dma.wait_for_transfer"(%43) : (!dma.token) -> ()
      "func.return"() : () -> ()
    }) {translation_info = #iree_codegen.translation_info<None>} : () -> ()
  }) : () -> ()
  "hal.executable.variant_end"() : () -> ()
}) {sym_name = "static", target = #hal.executable.target<"quidditch", "static", {compute_cores = 8 : i32, data_layout = "e-m:e-p:32:32-i64:64-n32-S128", target_triple = "riscv32-unknown-elf"}>} : () -> ()
```

## Or I modify configureForSnitch.cpp and get a different error:

From this file: /home/hoppip/Quidditch/codegen/compiler/src/Quidditch/Dialect/Snitch/Transforms/LowerL1Allocations.cpp

## What does transformed kernel look like when configureForSnitch is used?

```
  %c0 = arith.constant 0 : index
  %c400 = arith.constant 400 : index
  %c100 = arith.constant 100 : index
  %23 = scf.for %arg0 = %c0 to %c400 step %c100 iter_args(%arg1 = %22) -> (tensor<1x1200xf64>) {
    %c0_0 = arith.constant 0 : index
    %c1200 = arith.constant 1200 : index
    %c40 = arith.constant 40 : index
    %25 = scf.for %arg2 = %c0_0 to %c1200 step %c40 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
      %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, 100] [1, 1] : tensor<1x400xf64> to tensor<1x100xf64>
      %extracted_slice_1 = tensor.extract_slice %19[%arg2, %arg0] [40, 100] [1, 1] : tensor<1200x400xf64> to tensor<40x100xf64>
      %extracted_slice_2 = tensor.extract_slice %arg3[0, %arg2] [1, 40] [1, 1] : tensor<1x1200xf64> to tensor<1x40xf64>
      %26 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%extracted_slice, %extracted_slice_1 : tensor<1x100xf64>, tensor<40x100xf64>) outs(%extracted_slice_2 : tensor<1x40xf64>) -> tensor<1x40xf64>
      %inserted_slice = tensor.insert_slice %26 into %arg3[0, %arg2] [1, 40] [1, 1] : tensor<1x40xf64> into tensor<1x1200xf64>
      scf.yield %inserted_slice : tensor<1x1200xf64>
    }
    scf.yield %25 : tensor<1x1200xf64>
  }
```



Full output:

```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: This is the rewritten kernel!!!!!

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.extui %0 : i32 to i64
  %6 = arith.extui %1 : i32 to i64
  %7 = arith.shli %6, %c32_i64 : i64
  %8 = arith.ori %5, %7 : i64
  %9 = arith.index_castui %8 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %10 = arith.index_castui %2 : i32 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>>
  %16 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  %17 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %19 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1200, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>> -> tensor<1200x400xf64>
  %20 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %c0 = arith.constant 0 : index
  %c400 = arith.constant 400 : index
  %c100 = arith.constant 100 : index
  %23 = scf.for %arg0 = %c0 to %c400 step %c100 iter_args(%arg1 = %22) -> (tensor<1x1200xf64>) {
    %c0_0 = arith.constant 0 : index
    %c1200 = arith.constant 1200 : index
    %c40 = arith.constant 40 : index
    %25 = scf.for %arg2 = %c0_0 to %c1200 step %c40 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
      %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, 100] [1, 1] : tensor<1x400xf64> to tensor<1x100xf64>
      %extracted_slice_1 = tensor.extract_slice %19[%arg2, %arg0] [40, 100] [1, 1] : tensor<1200x400xf64> to tensor<40x100xf64>
      %extracted_slice_2 = tensor.extract_slice %arg3[0, %arg2] [1, 40] [1, 1] : tensor<1x1200xf64> to tensor<1x40xf64>
      %26 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%extracted_slice, %extracted_slice_1 : tensor<1x100xf64>, tensor<40x100xf64>) outs(%extracted_slice_2 : tensor<1x40xf64>) -> tensor<1x40xf64>
      %inserted_slice = tensor.insert_slice %26 into %arg3[0, %arg2] [1, 40] [1, 1] : tensor<1x40xf64> into tensor<1x1200xf64>
      scf.yield %inserted_slice : tensor<1x1200xf64>
    }
    scf.yield %25 : tensor<1x1200xf64>
  }
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %20 : tensor<1x1200xf64>, tensor<1x1200xf64>) outs(%17 : tensor<1x1200xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %25 = arith.addf %in, %in_0 : f64
    linalg.yield %25 : f64
  } -> tensor<1x1200xf64>
  flow.dispatch.tensor.store %24, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}
```



## old notes

old build output:
```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: i found a ZigZag input :) tilingScheme is [/home/hoppip/Quidditch/zigzag_tiling/grapeFruit/snitch-cluster-only-floats-no-ssrs-dispatch_1_matmul_transpose_b_1x1200x400_f64/grapeFruit-tiling-scheme.json]

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.extui %0 : i32 to i64
  %6 = arith.extui %1 : i32 to i64
  %7 = arith.shli %6, %c32_i64 : i64
  %8 = arith.ori %5, %7 : i64
  %9 = arith.index_castui %8 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %10 = arith.index_castui %2 : i32 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>>
  %16 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  %17 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %19 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1200, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>> -> tensor<1200x400xf64>
  %20 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %20 : tensor<1x1200xf64>, tensor<1x1200xf64>) outs(%17 : tensor<1x1200xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %25 = arith.addf %in, %in_0 : f64
    linalg.yield %25 : f64
  } -> tensor<1x1200xf64>
  flow.dispatch.tensor.store %24, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: zigzag parsed tilingScheme is [tiling scheme: {
bounds: [ [  1 ] [  16 ] [  5  8 ] ]
finalIndices: [ [  6 ] [  4 ] [  5  7 ] ]
}order: [ [  0  0 ] [  2  0 ] [  1  0 ] [  1  1 ] ]
}]

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.extui %0 : i32 to i64
  %6 = arith.extui %1 : i32 to i64
  %7 = arith.shli %6, %c32_i64 : i64
  %8 = arith.ori %5, %7 : i64
  %9 = arith.index_castui %8 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %10 = arith.index_castui %2 : i32 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>>
  %16 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  %17 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %19 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1200, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>> -> tensor<1200x400xf64>
  %20 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %20 : tensor<1x1200xf64>, tensor<1x1200xf64>) outs(%17 : tensor<1x1200xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %25 = arith.addf %in, %in_0 : f64
    linalg.yield %25 : f64
  } -> tensor<1x1200xf64>
  flow.dispatch.tensor.store %24, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}
```

## The long and winding road
Let's get our ZigZag Tiling Pass to tile one of the nsnet kernels!

Part of the output from `ninja -j 20` was
```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: i found a ZigZag input :) tilingScheme is [/home/hoppip/Quidditch/zigzag_tiling/zigzag-tile-scheme.json]

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.extui %0 : i32 to i64
  %6 = arith.extui %1 : i32 to i64
  %7 = arith.shli %6, %c32_i64 : i64
  %8 = arith.ori %5, %7 : i64
  %9 = arith.index_castui %8 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %10 = arith.index_castui %2 : i32 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>>
  %16 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  %17 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %19 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1200, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>> -> tensor<1200x400xf64>
  %20 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %20 : tensor<1x1200xf64>, tensor<1x1200xf64>) outs(%17 : tensor<1x1200xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %25 = arith.addf %in, %in_0 : f64
    linalg.yield %25 : f64
  } -> tensor<1x1200xf64>
  flow.dispatch.tensor.store %24, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: zigzag parsed tilingScheme is [tiling scheme: {
bounds: [ [  13 ] [  13 ] [  4  2 ] ]
finalIndices: [ [  3 ] [  4 ] [  6  5 ] ]
}order: [ [  2  0 ] [  2  1 ] [  1  0 ] [  0  0 ] ]
}]

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.extui %0 : i32 to i64
  %6 = arith.extui %1 : i32 to i64
  %7 = arith.shli %6, %c32_i64 : i64
  %8 = arith.ori %5, %7 : i64
  %9 = arith.index_castui %8 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %10 = arith.index_castui %2 : i32 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>>
  %16 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  %17 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %19 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1200, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>> -> tensor<1200x400xf64>
  %20 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %20 : tensor<1x1200xf64>, tensor<1x1200xf64>) outs(%17 : tensor<1x1200xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %25 = arith.addf %in, %in_0 : f64
    linalg.yield %25 : f64
  } -> tensor<1x1200xf64>
  flow.dispatch.tensor.store %24, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}
```

Function of interest:
```
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64()
```
Operation of interest within this function:
```
  %23 = linalg.matmul_transpose_b 
  {lowering_config = #quidditch_snitch.lowering_config<
                        l1_tiles = [0, 40, 100], 
                        l1_tiles_interchange = [2, 0, 1], 
                        dual_buffer = true>} 
    ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) 
    outs(%22 : tensor<1x1200xf64>) 
    -> tensor<1x1200xf64>
```
What does this operation look like as a linalg generic?

First, I wrapped the operation in a little function to plop into godbolt.org...
```
func.func @hacky_matmul_transpose_to_check_generic_form(
%18: tensor<1x400xf64>, %19: tensor<1200x400xf64>, %22: tensor<1x1200xf64>) -> (tensor<1x1200xf64>) {
  %23 = linalg.matmul_transpose_b 
  {lowering_config = #quidditch_snitch.lowering_config<
                        l1_tiles = [0, 40, 100], 
                        l1_tiles_interchange = [2, 0, 1], 
                        dual_buffer = true>} 
    ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) 
    outs(%22 : tensor<1x1200xf64>) 
    -> tensor<1x1200xf64>
    return %23 : tensor<1x1200xf64>
}
```
After running thru `mlir-opt` with the flags `-allow-unregistered-dialect -linalg-generalize-named-ops`,
```
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @hacky_matmul_transpose_to_check_generic_form(%arg0: tensor<1x400xf64>, %arg1: tensor<1200x400xf64>, %arg2: tensor<1x1200xf64>) -> tensor<1x1200xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%arg2 : tensor<1x1200xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<1x1200xf64>
    return %0 : tensor<1x1200xf64>
  }
}
```
Can I represent this linalg.generic as a ZigZag workload object?

Let's break this operation down, and then try to translate to a workload object...
```
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
%0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<1200x400xf64>) 
    outs(%arg2 : tensor<1x1200xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<1x1200xf64>

%in comes from %arg0 : tensor<1x400xf64> (using #map = affine_map<(d0, d1, d2) -> (d0, d2)>)
so d0 goes from 0 to 1, d2 goes from 0 to 400.
%in = %arg0[d0][d2]

%in_0 comes from %arg1 : tensor<1200x400xf64> (using #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>)
so d1 goes from 0 to 1200, d2 goes from 0 to 400.
%in_0 = %arg1[d1][d2]

%out comes from %arg2 : tensor<1x1200xf64> (using #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>)
so d0 goes from 0 to 1, d1 goes from 0 to 1200.
%out = %arg2[d0][d1]

Body of loop says %arg2[d0][d1] += %arg0[d0][d2] * %arg1[d1][d2]

Now let D0 = A, D1 = B, and D2 = C.

Body of loop says %arg2[a][b] += %arg0[a][c] * %arg1[b][c]

For ZigZag then, we say that
equation: O[a][b]+=I[a][c]*W[b][c]
loop_dims: [A,B,C]
loop_sizes: [1, 1200, 400]
```
ZigZag translation:
```
- id: 0 
  name: dispatch_1_matmul_transpose_b_1x1200x400_f64  # name can be used to specify mapping
  operator_type: default  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[b][c]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [1, 1200, 400]
  operand_precision:
    W: 64
    I: 64
    O: 64
    O_final: 64
  operand_source:
    I: 0
    W: 0
```
For a reality check, here is the function output from godbolt.org after `-one-shot-bufferize -convert-linalg-to-affine-loops`:
```
module {
  func.func @hacky_matmul_transpose_to_check_generic_form(%arg0: tensor<1x400xf64>, %arg1: tensor<1200x400xf64>, %arg2: tensor<1x1200xf64>) -> tensor<1x1200xf64> {
    %0 = bufferization.to_memref %arg1 : memref<1200x400xf64, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x400xf64, strided<[?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<1x1200xf64, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1200xf64>
    memref.copy %2, %alloc : memref<1x1200xf64, strided<[?, ?], offset: ?>> to memref<1x1200xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1200 {
        affine.for %arg5 = 0 to 400 {
          %4 = affine.load %1[%arg3, %arg5] : memref<1x400xf64, strided<[?, ?], offset: ?>>
          %5 = affine.load %0[%arg4, %arg5] : memref<1200x400xf64, strided<[?, ?], offset: ?>>
          %6 = affine.load %alloc[%arg3, %arg4] : memref<1x1200xf64>
          %7 = arith.mulf %4, %5 : f64
          %8 = arith.addf %6, %7 : f64
          affine.store %8, %alloc[%arg3, %arg4] : memref<1x1200xf64>
        }
      }
    }
    %3 = bufferization.to_tensor %alloc : memref<1x1200xf64>
    memref.dealloc %alloc : memref<1x1200xf64>
    return %3 : tensor<1x1200xf64>
  }
}
```
### Quick notes to delete later:
```
- id: 0 
  name: matmul_104_104  # name can be used to specify mapping
  operator_type: MatMul  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[c][b]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [104, 104, 104]
  operand_precision:
    W: 8
    I: 8
    O: 32
    O_final: 32
  operand_source:
    I: 0
    W: 0
```