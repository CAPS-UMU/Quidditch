#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
"builtin.module"() ({
// declaring external MLIR functions (implementations in C)
"func.func"() <{function_type = () -> index, sym_name = "myID", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
    index) // coreID
    -> (), sym_name = "wait_for_accelerator", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
"func.func"() <{function_type = () -> (), sym_name = "wait_for_all_accelerators", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
"func.func"() <{function_type = ( index,index,index,index,index,index,index,index) 
    -> (), sym_name = "save_outer_loop_counters", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
          index, 
          memref<104x104xi8, strided<[104, 1], offset: ?>>,  // input
          memref<26x104xi8, strided<[104, 1], offset: ?>>,   // weight slice
          memref<8x8xi32, strided<[8, 1], offset: ?>>)       // output slice
      -> (), sym_name = "dispatch_to_accelerator", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type =  (
  memref<2x16xi8>, 
  memref<16x2xi8, strided<[1,16]>>, 
  memref<2x2xi32, strided<[16,1]>>) 
  -> (), sym_name = "hola", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
"func.func"() <{function_type = (
  memref<104x13xi8, strided<[1, 104], offset: ?>>, 
  memref<104x13xi8, strided<[1, 104], offset: ?>>) 
  -> (), sym_name = "memrefCopy8bit", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

//print_memref_8_bit
"func.func"() <{function_type = (
  memref<26x104xi8, strided<[1, 104], offset: ?>>) 
  -> (), sym_name = "print_memref_8_bit", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
  memref<104x104xi32, strided<[8,1], offset: ?>>) 
  -> (), sym_name = "print_memref_32_bit", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (index) 
  -> (), sym_name = "print_index", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (i32) 
  -> (), sym_name = "print_i32", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
  memref<104x104xi8, strided<[104, 1], offset: ?>>, 
  memref<104x104xi8, strided<[104, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy8bit_I_104x104", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
"func.func"() <{function_type = (
  memref<104x104xi8, strided<[1, 104], offset: ?>>, 
  memref<104x104xi8, strided<[1, 104], offset: ?>>) 
  -> (), sym_name = "memrefCopy8bit_W_104x104", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
"func.func"() <{function_type = (
  memref<26x104xi8, strided<[104, 1], offset: ?>>, 
  memref<26x104xi8, strided<[104, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy8bit_W_26x104", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
"func.func"() <{function_type = (
  memref<8x8xi32, strided<[104, 1], offset: ?>>, 
  memref<8x8xi32, strided<[8, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy32bit_O_8x8_down", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
  memref<8x8xi32, strided<[8, 1], offset: ?>>, 
  memref<8x8xi32, strided<[104, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy32bit_O_8x8_up", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
  memref<104x1xi32, strided<[104, 1], offset: ?>>, 
  memref<104x1xi32, strided<[104, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy32bit", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
  memref<104x13xi32, strided<[104, 1], offset: ?>>, 
  memref<104x13xi32, strided<[104, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy32bit_O_104x13", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (
  memref<104x1xi32, strided<[104, 1], offset: ?>>, 
  memref<104x1xi32, strided<[104, 1], offset: ?>>) 
  -> (), sym_name = "memrefCopy32bit_O_104x1", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (index, index, i32) 
  -> (), sym_name = "print_weight_elt", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

"func.func"() <{function_type = (index, index, i32, index, index, index, index, index, index) 
  -> (), sym_name = "print_weight_elt2", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()


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
  }) {llvm.emit_c_interface}: () -> ()


"func.func"() <{function_type = 
    (index,                                             // coreID
    memref<104x104xi8, strided<[104, 1], offset: ?>>,   // input L3
    memref<104x104xi8, strided<[104, 1], offset: ?>>,   // weight L3
    memref<104x104xi32, strided<[104, 1], offset: ?>>,  // output L3
    memref<104x104xi8, strided<[104, 1], offset: ?>>,    // input L1
    memref<104x104xi8, strided<[104, 1], offset: ?>>,   // weight L1
    memref<8x8xi32, strided<[8, 1], offset: ?>>)    // output L1
    -> (), sym_name = "dmaCore"}> ({
  ^bb0(%coreID : index,
       %input: memref<104x104xi8, strided<[104, 1], offset: ?>>, 
       %weight: memref<104x104xi8, strided<[104,1], offset: ?>>, 
       %output: memref<104x104xi32, strided<[104,1], offset: ?>>, 
       %inputL1: memref<104x104xi8, strided<[104,1], offset: ?>>,
       %weightL1: memref<104x104xi8, strided<[104,1], offset: ?>>,
       %slice_O_L1: memref<8x8xi32, strided<[8,1], offset: ?>>): 

      // indices
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %four = arith.constant 4 : index
      %two = arith.constant  2 : index
      %eight = arith.constant  8 : index
      %thirteen = arith.constant 13 : index  
      // tile sizes
      %a1_bk_sz = arith.constant 8 : index
      %b1_bk_sz = arith.constant 8 : index
      %c1_bk_sz = arith.constant 13 : index
      %c2_bk_sz = arith.constant 26 : index
      %b0_bk_sz = arith.constant 1 : index
      %c0_bk_sz = arith.constant 1 : index
      %a0_bk_sz = arith.constant 1 : index
      // constants
      %sixTwentyFour = arith.constant 624: index
      %three = arith.constant 3: i8
      %nine = arith.constant 9: i8

      // copy input from L3 to L1
      func.call @memrefCopy8bit_I_104x104(%input,  %inputL1) 
      : (memref<104x104xi8, strided<[104, 1], offset: ?>>, memref<104x104xi8, strided<[104, 1], offset: ?>>) -> ()
     

      // FOR LOOPS START HERE
      scf.for %c2 = %zero to %four step %one iter_args() -> () {

      // select 26 x 104 subview of L3 weight matrix
      %slice_W_L3_offset = arith.muli %c2, %c2_bk_sz : index
      %slice_W_L3 = memref.subview %weight[%slice_W_L3_offset, %zero][26,104][1,1]
      :  memref<104x104xi8, strided<[104,1], offset: ?>> to memref<26x104xi8, strided<[104,1], offset: ?>>      
      
      // select 26 x 104 subview of L1 weight matrix
      %slice_W_L1 = memref.subview %weightL1[%zero, %zero][26,104][1,1] // is this offset correct?
      :  memref<104x104xi8, strided<[104,1], offset: ?>> to memref<26x104xi8, strided<[104,1], offset: ?>>
      
      // copy weights from L3 to L1
      func.call @memrefCopy8bit_W_26x104( %slice_W_L3, %slice_W_L1)
      : (memref<26x104xi8, strided<[104, 1], offset: ?>>,memref<26x104xi8, strided<[104, 1], offset: ?>>) -> ()

      scf.for %c1 = %zero to %two step %one iter_args() -> () {
      scf.for %b1 = %zero to %thirteen step %one iter_args() -> () {
      scf.for %a1 = %zero to %thirteen step %one iter_args() -> () {
      
      // select 8x8 subview of L3 output matrix
      %slice_O_L3_offset_r = arith.muli %a1, %a1_bk_sz : index
      %slice_O_L3_offset_c = arith.muli %b1, %b1_bk_sz : index
      %slice_O_L3 = memref.subview %output[%slice_O_L3_offset_r,%slice_O_L3_offset_c][8,8][1,1]
      : memref<104x104xi32, strided<[104, 1], offset: ?>>  to  memref<8x8xi32, strided<[104, 1], offset: ?>>
      
      // copy output tile 8x8 from L3 to L1.
      func.call @memrefCopy32bit_O_8x8_down(%slice_O_L3, %slice_O_L1)
      : (memref<8x8xi32, strided<[104, 1], offset: ?>>, memref<8x8xi32, strided<[8, 1], offset: ?>>) -> ()
      
      // after this point, the computation should be dispatched to compute cores.
      func.call @save_outer_loop_counters(%a1,%b1,%c1,%c2,%a1_bk_sz,%b1_bk_sz,%c1_bk_sz,%c2_bk_sz)
      : (index,index,index,index,index,index,index,index) -> ()

      scf.for %a0 = %zero to %eight step %one iter_args() -> () { // spatial unrolling loop

      func.call @dispatch_to_accelerator(%a0, %inputL1, %slice_W_L1, %slice_O_L1)
      : (
          index, 
          memref<104x104xi8, strided<[104, 1], offset: ?>>,  // input
          memref<26x104xi8, strided<[104, 1], offset: ?>>,   // weight slice
          memref<8x8xi32, strided<[8, 1], offset: ?>>)       // output slice
      -> ()
      } // end of spatial unrolling loop  

      func.call @wait_for_all_accelerators() : () -> ()

      // copy L1 output tile back to L3 
      func.call @memrefCopy32bit_O_8x8_up(%slice_O_L1, %slice_O_L3)
      : (memref<8x8xi32, strided<[8, 1], offset: ?>>, memref<8x8xi32, strided<[104, 1], offset: ?>>) -> ()      
      }}}  
      }      
  "func.return"() : () -> ()
  }) {llvm.emit_c_interface}: () -> ()

  // computation performed by accelerator as part of tiled matmul
  "func.func"() <{function_type = (
    memref<104x104xi8, strided<[104, 1], offset: ?>>,  // input
    memref<26x104xi8, strided<[104, 1], offset: ?>>,   // weight slice
    memref<8x8xi32, strided<[8, 1], offset: ?>>,       // output slice
    index,index,index,index,index,index,index,index)  
    -> (), sym_name = "computeCore"}> ({
  ^bb0(
    %slice_I_L1: memref<104x104xi8, strided<[104, 1], offset: ?>>, 
    %slice_W_L1: memref<26x104xi8, strided<[104, 1], offset: ?>>, 
    %slice_O_L1: memref<8x8xi32, strided<[8, 1], offset: ?>>,
    %a1 : index, 
    %b1 : index, 
    %c1 : index, 
    %c2 : index, 
    %a1_bk_sz_fooey : index, 
    %b1_bk_sz_fooey : index, 
    %c1_bk_sz_fooey : index, 
    %c2_bk_sz_fooey : index
    ):

    // indices
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %four = arith.constant 4 : index
      %two = arith.constant  2 : index
      %eight = arith.constant  8 : index
      %thirteen = arith.constant 13 : index  
      // tile sizes
      %a1_bk_sz = arith.constant 8 : index
      %b1_bk_sz = arith.constant 8 : index
      %c1_bk_sz = arith.constant 13 : index
      %c2_bk_sz = arith.constant 26 : index
      %b0_bk_sz = arith.constant 1 : index
      %c0_bk_sz = arith.constant 1 : index
      %a0_bk_sz = arith.constant 1 : index
      // constants
      %sixTwentyFour = arith.constant 624: index
      %three = arith.constant 3: i8
      %nine = arith.constant 9: i8

    %a0 = func.call @myID() : () -> (index) // spacial unrolling of a0
    scf.for %b0 = %zero to %eight step %one iter_args() -> () { 
    scf.for %c0 = %zero to %thirteen step %one iter_args() -> () {

      // index calculation
      %a_prod_0 = arith.muli %a0, %a0_bk_sz : index 
      %a_prod_1 = arith.muli %a1, %a1_bk_sz : index 
      %b_prod_0 = arith.muli %b0, %b0_bk_sz : index 
      %b_prod_1 = arith.muli %b1, %b1_bk_sz : index 
      %c_prod_0 = arith.muli %c0, %c0_bk_sz : index
      %c_prod_1 = arith.muli %c1, %c1_bk_sz : index
      %c_prod_2 = arith.muli %c2, %c2_bk_sz : index
      %c_sum_2_1 = arith.addi %c_prod_2, %c_prod_1 : index
      // indices into weight matrix slice
      %w_c = arith.addi %c_prod_1, %c_prod_0 : index 
      %w_b = arith.addi %b_prod_1, %b_prod_0 : index
      // indices into output matrix slice
      %o_a = arith.muli %a0, %a0_bk_sz : index 
      %o_b = arith.muli %b0, %b0_bk_sz : index 
      // indices into input matrix slice
      %i_a = arith.addi %a_prod_1, %a_prod_0 : index
      %i_c = arith.addi %c_sum_2_1, %c_prod_0 : index

      // load the input element
      %inputElt = memref.load %slice_I_L1[%i_a, %i_c] : memref<104x104xi8, strided<[104, 1], offset: ?>>
      %inputEltCasted = arith.extsi  %inputElt : i8 to i32 

      // load the weight element
      %weightElt = memref.load %slice_W_L1[%w_c, %w_b] : memref<26x104xi8, strided<[104,1], offset: ?>>
      %weightEltCasted = arith.extsi  %weightElt : i8 to i32 

      // load the output element
      %outputElt = memref.load %slice_O_L1[%o_a, %o_b] : memref<8x8xi32, strided<[8, 1], offset: ?>>  
      
      // perform the multiply-accumulate operation
      %prod = arith.muli %inputEltCasted, %weightEltCasted : i32
      %newOutputElt = arith.addi %prod, %outputElt : i32

      // update the L1 output tile with the newly computed value
      memref.store %newOutputElt, %slice_O_L1[%o_a, %o_b] : memref<8x8xi32, strided<[8, 1], offset: ?>>
    }
    }
    "func.return"() : () -> ()
  }) {llvm.emit_c_interface}: () -> ()




}) : () -> ()