# grapefruit notes
## main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64
As ZigZag workload:
```
- id: 0 
  name: dispatch_1_matmul_transpose_b_1x1200x400_f64  # name can be used to specify mapping
  operator_type: default  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[b][c]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [1, 400, 1200]
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

In my simplified json format, what would this tiling scheme look like?
```
hoodle
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
loop_sizes: [1, 400, 1200]
```
ZigZag translation:
```
- id: 0 
  name: dispatch_1_matmul_transpose_b_1x1200x400_f64  # name can be used to specify mapping
  operator_type: default  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[b][c]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [1, 400, 1200]
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