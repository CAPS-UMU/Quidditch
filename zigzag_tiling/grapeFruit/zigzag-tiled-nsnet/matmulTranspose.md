# Examining linalg.matmul_transpose_b

What does a `linalg.matmul_transpose_b` really look like?

```
mlir-opt --allow-unregistered-dialect --mlir-print-op-generic --linalg-generalize-named-ops matmulTranspose.mlir
```

Input:

```
func.func @matmultranspose(
%lhs: tensor<1x400xf64>, 
%rhs: tensor<1200x400xf64>, 
%acc: tensor<1x1200xf64>) -> tensor<1x1200xf64> {
  %24 = linalg.matmul_transpose_b 
  ins(%lhs, %rhs : tensor<1x400xf64>, tensor<1200x400xf64>) 
  outs(%acc : tensor<1x1200xf64>) -> tensor<1x1200xf64> 
 return %24 : tensor<1x1200xf64> 
}
```

Output:

```
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmultranspose(
  %arg0: tensor<1x400xf64>, 
  %arg1: tensor<1200x400xf64>, 
  %arg2: tensor<1x1200xf64>) -> tensor<1x1200xf64> {
    %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<1200x400xf64>) 
    outs(%arg2 : tensor<1x1200xf64>) {
    // %in = %arg0[d0, d2] ~ tensor<1x400xf64>[a,c]
    // %in_0 = %arg1[d1, d2] ~ tensor<1200x400xf64>[b,c]
    // %out = %arg2[d0, d1] ~ tensor<1x1200xf64>[a,b]
    // O[a,b] = I[a,c]*W[b,c]
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<1x1200xf64>
    return %0 : tensor<1x1200xf64>
  }
}
```

Lower to affine? `mlir-opt --linalg-generalize-named-ops -convert-linalg-to-affine-loops `

```
mlir-opt --linalg-generalize-named-ops --one-shot-bufferize -convert-linalg-to-affine-loops matmulTranspose.mlir
```

```
module {
  func.func @matmultranspose(%arg0: tensor<1x400xf64>, %arg1: tensor<1200x400xf64>, %arg2: tensor<1x1200xf64>) -> tensor<1x1200xf64> {
    %0 = bufferization.to_memref %arg1 : memref<1200x400xf64, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x400xf64, strided<[?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<1x1200xf64, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1200xf64>
    memref.copy %2, %alloc : memref<1x1200xf64, strided<[?, ?], offset: ?>> to memref<1x1200xf64>
    affine.for %arg3 = 0 to 1 {        // A
      affine.for %arg4 = 0 to 1200 {   // B
        affine.for %arg5 = 0 to 400 {  // C
          %4 = affine.load %1[%arg3, %arg5] // I[A][C]
          : memref<1x400xf64, strided<[?, ?], offset: ?>>
          %5 = affine.load %0[%arg4, %arg5] // W[B][C]
          : memref<1200x400xf64, strided<[?, ?], offset: ?>>
          %6 = affine.load %alloc[%arg3, %arg4] // O[A][B] 
          : memref<1x1200xf64>
          %7 = arith.mulf %4, %5 : f64
          %8 = arith.addf %6, %7 : f64
          affine.store %8, %alloc[%arg3, %arg4] : memref<1x1200xf64>
        }
      }
    } // So really, O[A][B] += I[A][C] * W[B][C]
    %3 = bufferization.to_tensor %alloc : memref<1x1200xf64>
    memref.dealloc %alloc : memref<1x1200xf64>
    return %3 : tensor<1x1200xf64>
  }
}
```

## Extra notes

Output (in generic syntax):

```
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<1x400xf64>, %arg1: tensor<1200x400xf64>, %arg2: tensor<1x1200xf64>):
    %0 = "linalg.generic"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):
      %1 = "arith.mulf"(%arg3, %arg4) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      %2 = "arith.addf"(%arg5, %1) {fastmath = #arith.fastmath<none>} : (f64, f64) -> f64
      "linalg.yield"(%2) : (f64) -> ()
    }) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<1x400xf64>, tensor<1200x400xf64>, tensor<1x1200xf64>) -> tensor<1x1200xf64>
    "func.return"(%0) : (tensor<1x1200xf64>) -> ()
  }) {function_type = (tensor<1x400xf64>, tensor<1200x400xf64>, tensor<1x1200xf64>) -> tensor<1x1200xf64>, sym_name = "matmultranspose"} : () -> ()
}) : () -> ()
```

