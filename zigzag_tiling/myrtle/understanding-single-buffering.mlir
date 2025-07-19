poodle and pug: 
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() 
attributes {translation_info = #iree_codegen.translation_info<None>} {
  %0 = quidditch_snitch.l1_memory_view -> memref<100000xi8>

  %c1200 = arith.constant 1200 : index
  %c96 = arith.constant 96 : index
  %c1248 = arith.constant 1248 : index
  %c40 = arith.constant 40 : index
  %c400 = arith.constant 400 : index
  %c0 = arith.constant 0 : index
  %c32_i64 = arith.constant 32 : i64

  %1 = hal.interface.constant.load[0] : i32
  %2 = hal.interface.constant.load[1] : i32
  %3 = hal.interface.constant.load[2] : i32
  %4 = hal.interface.constant.load[3] : i32
  %5 = hal.interface.constant.load[4] : i32

  %6 = arith.extui %1 : i32 to i64
  %7 = arith.extui %2 : i32 to i64
  %8 = arith.shli %7, %c32_i64 : i64
  %9 = arith.ori %6, %8 : i64

  %10 = arith.index_castui %9 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = arith.index_castui %5 : i32 to index
  
  // I think this is the input vector in L3...
  %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) 
  : memref<1x400xf64, strided<[400, 1], offset: ?>>
  memref.assume_alignment %14, 64 : memref<1x400xf64, strided<[400, 1], offset: ?>>
  
  // I think this is the weight matrix in L3...
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) 
  : memref<1200x400xf64, strided<[400, 1], offset: ?>>
  memref.assume_alignment %15, 1 : memref<1200x400xf64, strided<[400, 1], offset: ?>>
 
 // is this the input vector for the elementwise addition in L3?
  %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%12) flags(ReadOnly) 
  : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  memref.assume_alignment %16, 1 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  
  // output vector in L3 I think?
  %17 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%13) 
  : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  memref.assume_alignment %17, 1 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>

  // Allocation #1: allocate a buffer of 1x1200
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64, #quidditch_snitch.l1_encoding>

  // this line gives away that this function will be run ONLY on COMPUTE CORES!!! 
  // But then later a DMA call?! So before the DMA ops are separated to a DMA core specific func??
  %18 = quidditch_snitch.compute_core_index
  %19 = affine.apply affine_map<()[s0] -> (s0 * 150)>()[%18]
  scf.for %arg0 = %19 to %c1200 step %c1200 {

    %subview_4 = memref.subview %alloca[0, %arg0] [1, 150] [1, 1] 
    : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    quidditch_snitch.memref.microkernel(%subview_4) : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %cst = arith.constant 0.000000e+00 : f64
      linalg.fill ins(%cst : f64) outs(%arg1 : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>)
    }
    quidditch_snitch.microkernel_fence
  }

  // Allocation #2: allocate a buffer of size 1x1248
  %alloca_0 = memref.alloca() {alignment = 64 : i64} 
  : memref<1x1248xf64, #quidditch_snitch.l1_encoding>
  %subview = memref.subview %alloca_0[0, 0] [1, 1200] [1, 1] 
  : memref<1x1248xf64, #quidditch_snitch.l1_encoding> 
  to memref<1x1200xf64, strided<[1248, 1]>, #quidditch_snitch.l1_encoding> 
  
  // DMA x-fer: why is it here? 
  // we allocated a 1x1200 L1 buffer, set its contents to zero, 
  // and then we copy that buffer to another buffer padded to size 1x1248 that is ALSO in L1?
  // is it possible that this dma transfer gets removed/optimized out/ cleaned up??
  // search in vs code explorer for "StartTransferOp" for more info.
  // if there is no padding, do we still get a dma.start_transfer from 
  // the zero-ed out L1 output vector to the L1 output vector that is actually used?
  %20 = dma.start_transfer 
  from %alloca : memref<1x1200xf64, #quidditch_snitch.l1_encoding> 
  to %subview : memref<1x1200xf64, strided<[1248, 1]>, #quidditch_snitch.l1_encoding>

  dma.wait_for_transfer %20

  scf.for %arg0 = %c0 to %c400 step %c40 {
    %subview_4 = memref.subview %14[0, %arg0] [1, 40] [1, 1] 
    : memref<1x400xf64, strided<[400, 1], offset: ?>> to memref<1x40xf64, strided<[400, 1], offset: ?>>
    
    //Allocation #3 allocate a buffer of 1x40
    %alloca_5 = memref.alloca() {alignment = 64 : i64} 
    : memref<1x40xf64, #quidditch_snitch.l1_encoding>
    %cast_6 = memref.cast %alloca_5 : memref<1x40xf64, #quidditch_snitch.l1_encoding> to memref<1x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>
    
    // DMA x-fer: copy 1x40 slice of input vector in L3 to 1x40 slice in L1
    %24 = dma.start_transfer 
    from %subview_4 : memref<1x40xf64, strided<[400, 1], offset: ?>> 
    to %cast_6 : memref<1x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>
    dma.wait_for_transfer %24

    scf.for %arg1 = %c0 to %c1248 step %c96 {
      %25 = affine.min affine_map<(d0) -> (1200, d0 + 96)>(%arg1)
      %26 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%25, %arg1)

      %subview_7 = memref.subview %15[%arg1, %arg0] [%26, 40] [1, 1] : 
      memref<1200x400xf64, strided<[400, 1], offset: ?>> to 
      memref<?x40xf64, strided<[400, 1], offset: ?>>
      
      //Allocation #4: allocate a buffer of size 96x40
      %alloca_8 = memref.alloca() {alignment = 64 : i64} 
      : memref<96x40xf64, #quidditch_snitch.l1_encoding>

      %subview_9 = memref.subview %alloca_8[0, 0] [%26, 40] [1, 1] 
      : memref<96x40xf64, #quidditch_snitch.l1_encoding> 
      to memref<?x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>
      
      // DMA x-fer: copy a 96x40 slice of weight matrix in L3 to a slice in L1
      // QUESTION: why doesn't the weight matrix need to be padded???
      // Is it some kind of affine map trickery?
      %27 = dma.start_transfer 
      from %subview_7 : memref<?x40xf64, strided<[400, 1], offset: ?>> 
      to %subview_9 : memref<?x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>
      dma.wait_for_transfer %27

      %subview_10 = memref.subview %alloca_0[0, %arg1] [1, 96] [1, 1] 
      : memref<1x1248xf64, #quidditch_snitch.l1_encoding> 
      to memref<1x96xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>

      %28 = affine.apply affine_map<()[s0] -> (s0 * 12)>()[%18]
      scf.for %arg2 = %28 to %c96 step %c96 {
        %subview_11 = memref.subview %alloca_8[%arg2, 0] [12, 40] [1, 1] : memref<96x40xf64, #quidditch_snitch.l1_encoding> to memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %subview_12 = memref.subview %subview_10[0, %arg2] [1, 12] [1, 1] 
        : memref<1x96xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding> 
        to memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        quidditch_snitch.memref.microkernel(%alloca_5, %subview_11, %subview_12) : 
        memref<1x40xf64, #quidditch_snitch.l1_encoding>, 
        memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>, 
        memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
        ^bb0(%arg3: memref<1x40xf64, #quidditch_snitch.l1_encoding>, 
             %arg4: memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>, 
             %arg5: memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 96, 40], l1_tiles_interchange = [2, 0, 1], myrtleCost = [1382400, 12]>} 
          ins(%arg3, 
              %arg4 : memref<1x40xf64, #quidditch_snitch.l1_encoding>, memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>) 
         outs(%arg5 : memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>)
        }
        quidditch_snitch.microkernel_fence
      }
    }
  }

  // Allocation #5: allocate a buffer of size 1x1200 
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %cast = memref.cast %alloca_1 : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>

// DMA x-fer copy the entire 1x1200 input to addition in L3 to an equally-sized slice in L1
  %21 = dma.start_transfer 
  from %16 : memref<1x1200xf64, strided<[1200, 1], offset: ?>> 
  to %cast : memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  dma.wait_for_transfer %21

  // Allocation #6: allocate a buffer of size 1x1200
  %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %cast_3 = memref.cast %alloca_2 : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  
  // DMA x-fer copy the entire output vector in L3 to an equally sized slice in L1
  %22 = dma.start_transfer 
  from %17 : memref<1x1200xf64, strided<[1200, 1], offset: ?>> 
  to %cast_3 : memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  dma.wait_for_transfer %22

  scf.for %arg0 = %19 to %c1200 step %c1200 {
    
    // the input to to the elementwise addition is the output of the matmul transpose
    %subview_4 = memref.subview %subview[0, %arg0] [1, 150] [1, 1] 
    : memref<1x1200xf64, strided<[1248, 1]>, #quidditch_snitch.l1_encoding> 
    to memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    
    %subview_5 = memref.subview %alloca_1[0, %arg0] [1, 150] [1, 1] 
    : memref<1x1200xf64, #quidditch_snitch.l1_encoding> 
    to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    
    %subview_6 = memref.subview %alloca_2[0, %arg0] [1, 150] [1, 1] 
    : memref<1x1200xf64, #quidditch_snitch.l1_encoding> 
    to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    
    quidditch_snitch.memref.microkernel(%subview_4, %subview_5, %subview_6) : memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
    ^bb0(%arg1: memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, 
         %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, 
         %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2 : memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) outs(%arg3 : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) {
      ^bb0(%in: f64, %in_7: f64, %out: f64):
        %24 = arith.addf %in, %in_7 : f64
        linalg.yield %24 : f64
      }
    }
    quidditch_snitch.microkernel_fence
  }

  // DMA x-fer: copy result of addition L1 to output vector in L3
  %23 = dma.start_transfer 
  from %alloca_2 : memref<1x1200xf64, #quidditch_snitch.l1_encoding> 
  to %17 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  dma.wait_for_transfer %23
  return
}