poodle and pug: 
/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
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
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %18 = quidditch_snitch.compute_core_index
  %19 = affine.apply affine_map<()[s0] -> (s0 * 150)>()[%18]
  scf.for %arg0 = %19 to %c1200 step %c1200 {
    %subview = memref.subview %alloca[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    quidditch_snitch.memref.microkernel(%subview) : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %cst = arith.constant 0.000000e+00 : f64
      linalg.fill ins(%cst : f64) outs(%arg1 : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>)
    }
    quidditch_snitch.microkernel_fence
  }
  scf.for %arg0 = %c0 to %c400 step %c100 {
    %subview = memref.subview %14[0, %arg0] [1, 100] [1, 1] : memref<1x400xf64, strided<[400, 1], offset: ?>> to memref<1x100xf64, strided<[400, 1], offset: ?>>
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<1x100xf64, #quidditch_snitch.l1_encoding>
    %cast_4 = memref.cast %alloca_3 : memref<1x100xf64, #quidditch_snitch.l1_encoding> to memref<1x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
    %23 = dma.start_transfer from %subview : memref<1x100xf64, strided<[400, 1], offset: ?>> to %cast_4 : memref<1x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
    dma.wait_for_transfer %23
    %alloca_5 = memref.alloca() {alignment = 64 : i64} : memref<40x100xf64, #quidditch_snitch.l1_encoding>
    %alloca_6 = memref.alloca() {alignment = 64 : i64} : memref<40x100xf64, #quidditch_snitch.l1_encoding>
    %subview_7 = memref.subview %15[0, %arg0] [40, 100] [1, 1] : memref<1200x400xf64, strided<[400, 1], offset: ?>> to memref<40x100xf64, strided<[400, 1], offset: ?>>
    %cast_8 = memref.cast %alloca_5 : memref<40x100xf64, #quidditch_snitch.l1_encoding> to memref<40x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
    %24 = dma.start_transfer from %subview_7 : memref<40x100xf64, strided<[400, 1], offset: ?>> to %cast_8 : memref<40x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
    %25:2 = scf.for %arg1 = %c40 to %c1200 step %c40 iter_args(%arg2 = %alloca_5, %arg3 = %24) -> (memref<40x100xf64, #quidditch_snitch.l1_encoding>, !dma.token) {
      %subview_10 = memref.subview %15[%arg1, %arg0] [40, 100] [1, 1] : memref<1200x400xf64, strided<[400, 1], offset: ?>> to memref<40x100xf64, strided<[400, 1], offset: ?>>
      %27 = affine.apply affine_map<(d0) -> ((d0 floordiv 40) mod 2)>(%arg1)
      %28 = scf.index_switch %27 -> memref<40x100xf64, #quidditch_snitch.l1_encoding> 
      case 0 {
        scf.yield %alloca_5 : memref<40x100xf64, #quidditch_snitch.l1_encoding>
      }
      default {
        scf.yield %alloca_6 : memref<40x100xf64, #quidditch_snitch.l1_encoding>
      }
      %cast_11 = memref.cast %28 : memref<40x100xf64, #quidditch_snitch.l1_encoding> to memref<40x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
      %29 = dma.start_transfer from %subview_10 : memref<40x100xf64, strided<[400, 1], offset: ?>> to %cast_11 : memref<40x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
      %30 = affine.apply affine_map<(d0) -> (d0 - 40)>(%arg1)
      %subview_12 = memref.subview %alloca[0, %30] [1, 40] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x40xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      dma.wait_for_transfer %arg3
      %31 = affine.apply affine_map<()[s0] -> (s0 * 5)>()[%18]
      scf.for %arg4 = %31 to %c40 step %c40 {
        %subview_13 = memref.subview %arg2[%arg4, 0] [5, 100] [1, 1] : memref<40x100xf64, #quidditch_snitch.l1_encoding> to memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %subview_14 = memref.subview %subview_12[0, %arg4] [1, 5] [1, 1] : memref<1x40xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> to memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        quidditch_snitch.memref.microkernel(%alloca_3, %subview_13, %subview_14) : memref<1x100xf64, #quidditch_snitch.l1_encoding>, memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
        ^bb0(%arg5: memref<1x100xf64, #quidditch_snitch.l1_encoding>, %arg6: memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg7: memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true, myrtleCost = [1440000, 30]>} ins(%arg5, %arg6 : memref<1x100xf64, #quidditch_snitch.l1_encoding>, memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>) outs(%arg7 : memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>)
        }
        quidditch_snitch.microkernel_fence
      }
      scf.yield %28, %29 : memref<40x100xf64, #quidditch_snitch.l1_encoding>, !dma.token
    }
    %subview_9 = memref.subview %alloca[0, 1160] [1, 40] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x40xf64, strided<[1200, 1], offset: 1160>, #quidditch_snitch.l1_encoding>
    dma.wait_for_transfer %25#1
    %26 = affine.apply affine_map<()[s0] -> (s0 * 5)>()[%18]
    scf.for %arg1 = %26 to %c40 step %c40 {
      %subview_10 = memref.subview %25#0[%arg1, 0] [5, 100] [1, 1] : memref<40x100xf64, #quidditch_snitch.l1_encoding> to memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      %subview_11 = memref.subview %subview_9[0, %arg1] [1, 5] [1, 1] : memref<1x40xf64, strided<[1200, 1], offset: 1160>, #quidditch_snitch.l1_encoding> to memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      quidditch_snitch.memref.microkernel(%alloca_3, %subview_10, %subview_11) : memref<1x100xf64, #quidditch_snitch.l1_encoding>, memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
      ^bb0(%arg2: memref<1x100xf64, #quidditch_snitch.l1_encoding>, %arg3: memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg4: memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
        linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1], dual_buffer = true, myrtleCost = [1440000, 30]>} ins(%arg2, %arg3 : memref<1x100xf64, #quidditch_snitch.l1_encoding>, memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>) outs(%arg4 : memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>)
      }
      quidditch_snitch.microkernel_fence
    }
  }
  %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %cast = memref.cast %alloca_0 : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %20 = dma.start_transfer from %16 : memref<1x1200xf64, strided<[1200, 1], offset: ?>> to %cast : memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  dma.wait_for_transfer %20
  %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %cast_2 = memref.cast %alloca_1 : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %21 = dma.start_transfer from %17 : memref<1x1200xf64, strided<[1200, 1], offset: ?>> to %cast_2 : memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  dma.wait_for_transfer %21
  scf.for %arg0 = %19 to %c1200 step %c1200 {
    %subview = memref.subview %alloca[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %subview_3 = memref.subview %alloca_0[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %subview_4 = memref.subview %alloca_1[0, %arg0] [1, 150] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    quidditch_snitch.memref.microkernel(%subview, %subview_3, %subview_4) : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding> {
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2 : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) outs(%arg3 : memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) {
      ^bb0(%in: f64, %in_5: f64, %out: f64):
        %23 = arith.addf %in, %in_5 : f64
        linalg.yield %23 : f64
      }
    }
    quidditch_snitch.microkernel_fence
  }
  %22 = dma.start_transfer from %alloca_1 : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to %17 : memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  dma.wait_for_transfer %22
  return
}