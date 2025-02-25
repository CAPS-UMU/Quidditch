# Matrix Multiplication 6

- This example runs tiled matrix multiplication on a snitch compute core.
- Tiling scheme is chosen by ZigZag using gemm.py hardware description.
- Dynamically Allocated Input
- Matrix size 104 x 104
- All data on L3

[back to all tests](../../../zigzag-fork/README.md#Examples)

## I. Input to ZigZag

#### a. MLIR (use linalg-to-stream tool)

python xdsl_opt_main.py tests/matmul6.mlir -p linalg-to-stream; python run_zigzag.py

```
// matrices are 104 x 104
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
"builtin.module"() ({
 
"func.func"() <{function_type = (memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, memref<104x104xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5)  : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6)  : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4)  : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, i32, i32, memref<104x104xi32>) -> ()
    "func.return"() : () -> ()
  }) {llvm.emit_c_interface} : () -> ()
  }) : () -> ()
```

#### b. Equivalent Python Workload Object (directly passed to zigzag)

```
workload = {
    0: {
        "operator_type": "default",
        "equation": "O[d0][d1] += I[d0][d2] * W[d2][d1]",
        "dimension_relations": [],
        "loop_dim_size": {"D0": 104, "D1": 104, "D2": 104},
        "operand_precision": {"O": 32, "O_final": 32, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
        "padding": {"D0": (0, 0), "D2": (0, 0)},
    }
}
```

#### c. C code Equivalent (cannot feed to zigzag; just for reference)

C-ish pseudocode (ignoring sign extension and subtracting 0 instructions)

```
for d0; d0 < 104; d0++:
for d1; d1 < 104; d1++;
for d2; d2 < 104; d2++;
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
```

#### d. SNAX Gemm Accelerator Hardware Description

- [This is the hardware description fed to ZigZag](https://github.com/xdslproject/linalg-to-stream/blob/main/inputs/hardware/snax_gemm.py)

- Eventually we want a snitch cluster hardware description, because that is the actual target we are running on in this repo

## II. Output from ZigZag

```
========================================================================
Temporal Loops                   O            W            I            
========================================================================
for D0 in [0, 13):               l3           l1           l1           
------------------------------------------------------------------------
  for D1 in [0, 13):             l1           l1           l1           
------------------------------------------------------------------------
    for D2 in [0, 13):           rf_32b_O     l1           l1           
------------------------------------------------------------------------
========================================================================
Spatial Loops                                                           
========================================================================
      parfor D0 in [0, 8):                                              
------------------------------------------------------------------------
      parfor D1 in [0, 8):                                              
------------------------------------------------------------------------
      parfor D2 in [0, 8):                                              
------------------------------------------------------------------------
```

## III. Manual Transformation

#### a. C code transformed

```
void matmul_transformed(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y,
                               TwoDMemrefI32_t *z) {
  // only square matrices allowed
  size_t d0_1_bk_sz = MAT_WIDTH / 13;  // 8
  size_t d1_1_bk_sz = MAT_WIDTH / 13;  // 8
  size_t d2_1_bk_sz = MAT_WIDTH / 13;  // 8
   
  for (size_t d0_1 = 0; d0_1 < 13; d0_1++) { // 13 8-elt-row chunks
  
  // everything inside here should be dispatched to the accelerator (I think)
   for (size_t d1_1 = 0; d1_1 < 13; d1_1++) {
    for (size_t d2_1 = 0; d2_1 < 13; d2_1++) {
    
     // these inner three loops should be spacially unrolled, but ignore for now...
     for (size_t d0_2 = 0; d0_2 < 8; d0_2++) {
      for (size_t d1_2 = 0; d1_2 < 8; d1_2++) {
       for (size_t d2_2 = 0; d2_2 < 8; d2_2++) {
       
        // calculate indices
        size_t d0 = d0_1 * d0_1_bk_sz + d0_2;
        size_t d1 = d1_1 * d1_1_bk_sz + d1_2;
        size_t d2 = d2_1 * d2_1_bk_sz + d2_2;
        
        //perform MAC
        z_index = (d0 * MAT_WIDTH) + d1;
        x_index = (d0 * MAT_WIDTH) + d2;
        y_index = (d2 * MAT_WIDTH) + d1;
        z->aligned_data[z_index] +=
            x->aligned_data[x_index] * y->aligned_data[y_index];
       }
      }
     }
    }
   }
  }                           
}
```

#### b. MLIR transformed

```
  "func.func"() <{function_type = (memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, memref<104x104xi32, strided<[104,1]>>) -> (), sym_name = "tiled_matmul"}> ({
  ^bb0(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1,104]>>, %arg2: memref<104x104xi32, strided<[104,1]>>):
    // tile sizes
    %d0_1_bk_sz = arith.constant 8 : index
    %d1_1_bk_sz = arith.constant 8 : index
    %d2_1_bk_sz = arith.constant 8 : index
    
    // indices
    %zero = arith.constant 0 : index
    %one = arith.constant 1: index
    %eight = arith.constant 8 : index
    %thirteen = arith.constant 13 : index

    // enter scf nested FOR LOOP
    scf.for %d0_1 = %zero to %thirteen step %one iter_args() -> () {
    scf.for %d1_1 = %zero to %thirteen step %one iter_args() -> () {  
    scf.for %d2_1 = %zero to %thirteen step %one iter_args () -> () {
    scf.for %d0_2 = %zero to %eight step %one iter_args() -> () {       
    scf.for %d1_2 = %zero to %eight step %one iter_args () -> () { 
    scf.for %d2_2 = %zero to %eight step %one iter_args () -> () {
      %prod0 = arith.muli %d0_1, %d0_1_bk_sz : index
      %d0 = arith.addi %prod0, %d0_2 : index

      %prod1 = arith.muli %d1_1, %d1_1_bk_sz : index
      %d1 = arith.addi %prod1, %d1_2 : index

      %prod2 = arith.muli %d2_1, %d2_1_bk_sz : index
      %d2 = arith.addi %prod2, %d2_2 : index

      %inputElt = memref.load %arg0[%d0, %d2] : memref<104x104xi8>
      %inputEltCasted = arith.extsi  %inputElt : i8 to i32 
      
      %weightElt = memref.load %arg1[%d2, %d1] : memref<104x104xi8, strided<[1,104]>>
      %weightEltCasted = arith.extsi  %weightElt : i8 to i32 

      %outputElt = memref.load %arg2[%d0, %d1] : memref<104x104xi32, strided<[104,1]>> 
      %prod = arith.muli %inputEltCasted, %weightEltCasted : i32

      %newOutputElt = arith.addi %prod, %outputElt : i32 
      memref.store %newOutputElt, %arg2[%d0, %d1] : memref<104x104xi32, strided<[104,1]>>

    } // end of d2_2 for
    } // end of d1_2 for
    } // end of d0_2 for
    } // end of d2_1 for
    } // end of d1_1 for
    } // end of d0_1 for

    "func.return"() : () -> ()
  }) {llvm.emit_c_interface}: () -> ()
```

## IV. Running the transformed MLIR on Snitch

```
cd runtime/tests
```

spike: 

```
sh zigzag-spike-build-and-run.sh tiledMatmul6.mlir
```

verilator:

```
sh zigzag-verilator-build-and-run.sh tiledMatmul6.mlir
```

## V. Expected Output

```
[hoppip@inf-205-4 tests]$ sh zigzag-spike-build-and-run.sh tiledMatmul6.mlir
tiledMatmul6
TiledMatmul6
rm: cannot remove 'tiledMatmul6/out/*.svg': No such file or directory
warning: overriding the module target triple with riscv32-unknown-unknown-elf [-Woverride-module]
1 warning generated.
-- Could not find nvcc, please set CUDAToolkit_ROOT.
-- IREE HAL drivers:
-- IREE HAL local executable library loaders:
-- IREE HAL local executable plugin mechanisms:
-- Not adding LLVM/MLIR because the configuration does not require it
-- Disabling compiler loader (platform does not support library loading)
-- Configuring IREE runtime plugins
-- Runtime tracing disabled
-- Configuring done (0.3s)
-- Generating done (0.1s)
-- Build files have been written to: /home/hoppip/Quidditch-zigzag/runtime/build
[0/2] Re-checking globbed directories...
[1/1] Linking C executable tests/TiledMatmul6
Output Correct
```

