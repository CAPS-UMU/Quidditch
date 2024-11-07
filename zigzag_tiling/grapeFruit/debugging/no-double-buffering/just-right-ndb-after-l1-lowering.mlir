<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Turnip EVERYTHING is over!!

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %0 = quidditch_snitch.l1_memory_view -> memref<100000xi8>
  %c40 = arith.constant 40 : index
  %c1200 = arith.constant 1200 : index
  %c100 = arith.constant 100 : index
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
  %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : memref<1x400xf64, strided<[400, 1], offset: ?>>
  memref.assume_alignment %14, 64 : memref<1x400xf64, strided<[400, 1], offset: ?>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : memref<1200x400xf64, strided<[400, 1], offset: ?>>
  memref.assume_alignment %15, 1 : memref<1200x400xf64, strided<[400, 1], offset: ?>>
  %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%12) flags(ReadOnly) : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  memref.assume_alignment %16, 1 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  %17 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%13) : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  memref.assume_alignment %17, 1 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  %c0_0 = arith.constant 0 : index
  %view = memref.view %0[%c0_0][] : memref<100000xi8> to memref<1200xf64>
  %reinterpret_cast = memref.reinterpret_cast %view to offset: [0], sizes: [1, 1200], strides: [1200, 1] : memref<1200xf64> to memref<1x1200xf64>
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64>
  %18 = quidditch_snitch.compute_core_index
  %19 = affine.apply affine_map<()[s0] -> (s0 * 150)>()[%18]
  scf.for %arg0 = %19 to %c1200 step %c1200 {
    %subview = memref.subview %reinterpret_cast[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64> to memref<1x150xf64, strided<[1200, 1], offset: ?>>
    quidditch_snitch.memref.microkernel(%subview) : memref<1x150xf64, strided<[1200, 1], offset: ?>> {
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>>):
      %cst = arith.constant 0.000000e+00 : f64
      linalg.fill ins(%cst : f64) outs(%arg1 : memref<1x150xf64, strided<[1200, 1], offset: ?>>)
    }
    quidditch_snitch.microkernel_fence
  }
  scf.for %arg0 = %c0 to %c400 step %c100 {
    %subview = memref.subview %14[0, %arg0] [1, 100] [1, 1] : memref<1x400xf64, strided<[400, 1], offset: ?>> to memref<1x100xf64, strided<[400, 1], offset: ?>>
    %c9600 = arith.constant 9600 : index
    %view_8 = memref.view %0[%c9600][] : memref<100000xi8> to memref<100xf64>
    %reinterpret_cast_9 = memref.reinterpret_cast %view_8 to offset: [0], sizes: [1, 100], strides: [100, 1] : memref<100xf64> to memref<1x100xf64>
    %alloca_10 = memref.alloca() {alignment = 64 : i64} : memref<1x100xf64>
    %cast_11 = memref.cast %reinterpret_cast_9 : memref<1x100xf64> to memref<1x100xf64, strided<[100, 1]>>
    %23 = dma.start_transfer from %subview : memref<1x100xf64, strided<[400, 1], offset: ?>> to %cast_11 : memref<1x100xf64, strided<[100, 1]>>
    dma.wait_for_transfer %23
    scf.for %arg1 = %c0 to %c1200 step %c40 {
      %subview_12 = memref.subview %15[%arg1, %arg0] [40, 100] [1, 1] : memref<1200x400xf64, strided<[400, 1], offset: ?>> to memref<40x100xf64, strided<[400, 1], offset: ?>>
      %subview_13 = memref.subview %reinterpret_cast[0, %arg1] [1, 40] [1, 1] : memref<1x1200xf64> to memref<1x40xf64, strided<[1200, 1], offset: ?>>
      %c10432 = arith.constant 10432 : index
      %view_14 = memref.view %0[%c10432][] : memref<100000xi8> to memref<4000xf64>
      %reinterpret_cast_15 = memref.reinterpret_cast %view_14 to offset: [0], sizes: [40, 100], strides: [100, 1] : memref<4000xf64> to memref<40x100xf64>
      %alloca_16 = memref.alloca() {alignment = 64 : i64} : memref<40x100xf64>
      %cast_17 = memref.cast %reinterpret_cast_15 : memref<40x100xf64> to memref<40x100xf64, strided<[100, 1]>>
      %24 = dma.start_transfer from %subview_12 : memref<40x100xf64, strided<[400, 1], offset: ?>> to %cast_17 : memref<40x100xf64, strided<[100, 1]>>
      dma.wait_for_transfer %24
      %25 = affine.apply affine_map<()[s0] -> (s0 * 5)>()[%18]
      scf.for %arg2 = %25 to %c40 step %c40 {
        %subview_18 = memref.subview %reinterpret_cast_15[%arg2, 0] [5, 100] [1, 1] : memref<40x100xf64> to memref<5x100xf64, strided<[100, 1], offset: ?>>
        %subview_19 = memref.subview %subview_13[0, %arg2] [1, 5] [1, 1] : memref<1x40xf64, strided<[1200, 1], offset: ?>> to memref<1x5xf64, strided<[1200, 1], offset: ?>>
        quidditch_snitch.memref.microkernel(%reinterpret_cast_9, %subview_18, %subview_19) : memref<1x100xf64>, memref<5x100xf64, strided<[100, 1], offset: ?>>, memref<1x5xf64, strided<[1200, 1], offset: ?>> {
        ^bb0(%arg3: memref<1x100xf64>, %arg4: memref<5x100xf64, strided<[100, 1], offset: ?>>, %arg5: memref<1x5xf64, strided<[1200, 1], offset: ?>>):
          linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1]>} ins(%arg3, %arg4 : memref<1x100xf64>, memref<5x100xf64, strided<[100, 1], offset: ?>>) outs(%arg5 : memref<1x5xf64, strided<[1200, 1], offset: ?>>)
        }
        quidditch_snitch.microkernel_fence
      }
    }
  }
  %c42432 = arith.constant 42432 : index
  %view_1 = memref.view %0[%c42432][] : memref<100000xi8> to memref<1200xf64>
  %reinterpret_cast_2 = memref.reinterpret_cast %view_1 to offset: [0], sizes: [1, 1200], strides: [1200, 1] : memref<1200xf64> to memref<1x1200xf64>
  %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64>
  %cast = memref.cast %reinterpret_cast_2 : memref<1x1200xf64> to memref<1x1200xf64, strided<[1200, 1]>>
  %20 = dma.start_transfer from %16 : memref<1x1200xf64, strided<[1200, 1], offset: ?>> to %cast : memref<1x1200xf64, strided<[1200, 1]>>
  dma.wait_for_transfer %20
  %c52032 = arith.constant 52032 : index
  %view_4 = memref.view %0[%c52032][] : memref<100000xi8> to memref<1200xf64>
  %reinterpret_cast_5 = memref.reinterpret_cast %view_4 to offset: [0], sizes: [1, 1200], strides: [1200, 1] : memref<1200xf64> to memref<1x1200xf64>
  %alloca_6 = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64>
  %cast_7 = memref.cast %reinterpret_cast_5 : memref<1x1200xf64> to memref<1x1200xf64, strided<[1200, 1]>>
  %21 = dma.start_transfer from %17 : memref<1x1200xf64, strided<[1200, 1], offset: ?>> to %cast_7 : memref<1x1200xf64, strided<[1200, 1]>>
  dma.wait_for_transfer %21
  scf.for %arg0 = %19 to %c1200 step %c1200 {
    %subview = memref.subview %reinterpret_cast[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64> to memref<1x150xf64, strided<[1200, 1], offset: ?>>
    %subview_8 = memref.subview %reinterpret_cast_2[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64> to memref<1x150xf64, strided<[1200, 1], offset: ?>>
    %subview_9 = memref.subview %reinterpret_cast_5[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64> to memref<1x150xf64, strided<[1200, 1], offset: ?>>
    quidditch_snitch.memref.microkernel(%subview, %subview_8, %subview_9) : memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>> {
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>>):
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2 : memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>) outs(%arg3 : memref<1x150xf64, strided<[1200, 1], offset: ?>>) {
      ^bb0(%in: f64, %in_10: f64, %out: f64):
        %23 = arith.addf %in, %in_10 : f64
        linalg.yield %23 : f64
      }
    }
    quidditch_snitch.microkernel_fence
  }
  %22 = dma.start_transfer from %reinterpret_cast_5 : memref<1x1200xf64> to %17 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  dma.wait_for_transfer %22
  return
}