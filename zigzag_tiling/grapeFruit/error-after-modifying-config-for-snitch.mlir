<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: kernel does not fit into L1 memory and cannot be compiled
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
"func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
  %0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>
  %1 = "arith.constant"() <{value = 32 : index}> : () -> index
  %2 = "arith.constant"() <{value = 25 : index}> : () -> index
  %3 = "arith.constant"() <{value = 1200 : index}> : () -> index
  %4 = "arith.constant"() <{value = 240 : index}> : () -> index
  %5 = "arith.constant"() <{value = 480 : index}> : () -> index
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
  %28 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %29 = "quidditch_snitch.compute_core_index"() : () -> index
  %30 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 150)>}> : (index) -> index
  "scf.for"(%30, %3, %3) ({
  ^bb0(%arg25: index):
    %86 = "memref.subview"(%27, %arg25) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%86) ({
    ^bb0(%arg26: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %87 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      "linalg.fill"(%87, %arg26) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg27: f64, %arg28: f64):
        "linalg.yield"(%arg27) : (f64) -> ()
      }) : (f64, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%6, %5, %4) ({
  ^bb0(%arg7: index):
    %42 = "affine.min"(%arg7) <{map = affine_map<(d0) -> (400, d0 + 240)>}> : (index) -> index
    %43 = "affine.apply"(%42, %arg7) <{map = affine_map<(d0, d1) -> (d0 - d1)>}> : (index, index) -> index
    %44 = "memref.subview"(%21, %arg7, %43) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<1x?xf64, strided<[400, 1], offset: ?>>
    %45 = "arith.constant"() <{value = 9600 : index}> : () -> index
    %46 = "memref.view"(%0, %45) : (memref<100000xi8>, index) -> memref<240xf64>
    %47 = "memref.reinterpret_cast"(%46) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 240>, static_strides = array<i64: 240, 1>}> : (memref<240xf64>) -> memref<1x240xf64>
    %48 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x240xf64, #quidditch_snitch.l1_encoding>
    %49 = "dma.start_zero_mem_transfer"(%47) : (memref<1x240xf64>) -> !dma.token
    "dma.wait_for_transfer"(%49) : (!dma.token) -> ()
    %50 = "memref.subview"(%47, %43) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x240xf64>, index) -> memref<1x?xf64, strided<[240, 1]>, #quidditch_snitch.l1_encoding>
    %51 = "dma.start_transfer"(%44, %50) : (memref<1x?xf64, strided<[400, 1], offset: ?>>, memref<1x?xf64, strided<[240, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
    "dma.wait_for_transfer"(%51) : (!dma.token) -> ()
    %52 = "arith.constant"() <{value = 11520 : index}> : () -> index
    %53 = "memref.view"(%0, %52) : (memref<100000xi8>, index) -> memref<6000xf64>
    %54 = "memref.reinterpret_cast"(%53) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 25, 240>, static_strides = array<i64: 240, 1>}> : (memref<6000xf64>) -> memref<25x240xf64>
    %55 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<25x240xf64, #quidditch_snitch.l1_encoding>
    %56 = "arith.constant"() <{value = 59520 : index}> : () -> index
    %57 = "memref.view"(%0, %56) : (memref<100000xi8>, index) -> memref<6000xf64>
    %58 = "memref.reinterpret_cast"(%57) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 25, 240>, static_strides = array<i64: 240, 1>}> : (memref<6000xf64>) -> memref<25x240xf64>
    %59 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<25x240xf64, #quidditch_snitch.l1_encoding>
    %60 = "memref.subview"(%22, %arg7, %43) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<25x?xf64, strided<[400, 1], offset: ?>>
    %61 = "dma.start_zero_mem_transfer"(%54) : (memref<25x240xf64>) -> !dma.token
    "dma.wait_for_transfer"(%61) : (!dma.token) -> ()
    %62 = "memref.subview"(%54, %43) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<25x240xf64>, index) -> memref<25x?xf64, strided<[240, 1]>, #quidditch_snitch.l1_encoding>
    %63 = "dma.start_transfer"(%60, %62) : (memref<25x?xf64, strided<[400, 1], offset: ?>>, memref<25x?xf64, strided<[240, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
    %64:2 = "scf.for"(%2, %3, %2, %54, %63) ({
    ^bb0(%arg15: index, %arg16: memref<25x240xf64, #quidditch_snitch.l1_encoding>, %arg17: !dma.token):
      %72 = "memref.subview"(%22, %arg15, %arg7, %43) <{operandSegmentSizes = array<i32: 1, 2, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index, index) -> memref<25x?xf64, strided<[400, 1], offset: ?>>
      %73 = "affine.apply"(%arg15) <{map = affine_map<(d0) -> ((d0 floordiv 25) mod 2)>}> : (index) -> index
      %74 = "scf.index_switch"(%73) <{cases = array<i64: 0>}> ({
        "scf.yield"(%58) : (memref<25x240xf64>) -> ()
      }, {
        "scf.yield"(%54) : (memref<25x240xf64>) -> ()
      }) : (index) -> memref<25x240xf64, #quidditch_snitch.l1_encoding>
      %75 = "dma.start_zero_mem_transfer"(%74) : (memref<25x240xf64, #quidditch_snitch.l1_encoding>) -> !dma.token
      "dma.wait_for_transfer"(%75) : (!dma.token) -> ()
      %76 = "memref.subview"(%74, %43) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<25x240xf64, #quidditch_snitch.l1_encoding>, index) -> memref<25x?xf64, strided<[240, 1]>, #quidditch_snitch.l1_encoding>
      %77 = "dma.start_transfer"(%72, %76) : (memref<25x?xf64, strided<[400, 1], offset: ?>>, memref<25x?xf64, strided<[240, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
      %78 = "affine.apply"(%arg15) <{map = affine_map<(d0) -> (d0 - 25)>}> : (index) -> index
      %79 = "memref.subview"(%27, %78) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 25>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x25xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      "dma.wait_for_transfer"(%arg17) : (!dma.token) -> ()
      %80 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 4)>}> : (index) -> index
      "scf.for"(%80, %2, %1) ({
      ^bb0(%arg18: index):
        %81 = "affine.min"(%arg18) <{map = affine_map<(d0) -> (-d0 + 25, 4)>}> : (index) -> index
        %82 = "memref.subview"(%arg16, %arg18, %81) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: -9223372036854775808, 240>, static_strides = array<i64: 1, 1>}> : (memref<25x240xf64, #quidditch_snitch.l1_encoding>, index, index) -> memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %83 = "memref.subview"(%79, %arg18, %81) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x25xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, index, index) -> memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        "quidditch_snitch.memref.microkernel"(%47, %82, %83) ({
        ^bb0(%arg19: memref<1x240xf64, #quidditch_snitch.l1_encoding>, %arg20: memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg21: memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          "linalg.matmul_transpose_b"(%arg19, %arg20, %arg21) <{operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg22: f64, %arg23: f64, %arg24: f64):
            %84 = "arith.mulf"(%arg22, %arg23) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            %85 = "arith.addf"(%arg24, %84) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%85) : (f64) -> ()
          }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 25, 240], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} : (memref<1x240xf64, #quidditch_snitch.l1_encoding>, memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        }) : (memref<1x240xf64>, memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"(%74, %77) : (memref<25x240xf64, #quidditch_snitch.l1_encoding>, !dma.token) -> ()
    }) : (index, index, index, memref<25x240xf64>, !dma.token) -> (memref<25x240xf64, #quidditch_snitch.l1_encoding>, !dma.token)
    %65 = "memref.subview"(%27) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 1175>, static_sizes = array<i64: 1, 25>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>) -> memref<1x25xf64, strided<[1200, 1], offset: 1175>, #quidditch_snitch.l1_encoding>
    "dma.wait_for_transfer"(%64#1) : (!dma.token) -> ()
    %66 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 4)>}> : (index) -> index
    "scf.for"(%66, %2, %1) ({
    ^bb0(%arg8: index):
      %67 = "affine.min"(%arg8) <{map = affine_map<(d0) -> (-d0 + 25, 4)>}> : (index) -> index
      %68 = "memref.subview"(%64#0, %arg8, %67) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: -9223372036854775808, 240>, static_strides = array<i64: 1, 1>}> : (memref<25x240xf64, #quidditch_snitch.l1_encoding>, index, index) -> memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      %69 = "memref.subview"(%65, %arg8, %67) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x25xf64, strided<[1200, 1], offset: 1175>, #quidditch_snitch.l1_encoding>, index, index) -> memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      "quidditch_snitch.memref.microkernel"(%47, %68, %69) ({
      ^bb0(%arg9: memref<1x240xf64, #quidditch_snitch.l1_encoding>, %arg10: memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg11: memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
        "linalg.matmul_transpose_b"(%arg9, %arg10, %arg11) <{operandSegmentSizes = array<i32: 2, 1>}> ({
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
          %70 = "arith.mulf"(%arg12, %arg13) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          %71 = "arith.addf"(%arg14, %70) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          "linalg.yield"(%71) : (f64) -> ()
        }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 25, 240], l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} : (memref<1x240xf64, #quidditch_snitch.l1_encoding>, memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
      }) : (memref<1x240xf64>, memref<?x240xf64, strided<[240, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x?xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
      "quidditch_snitch.microkernel_fence"() : () -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %31 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %32 = "memref.cast"(%31) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %33 = "dma.start_transfer"(%23, %32) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%33) : (!dma.token) -> ()
  %34 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %35 = "memref.cast"(%34) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %36 = "dma.start_transfer"(%24, %35) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%36) : (!dma.token) -> ()
  "scf.for"(%30, %3, %3) ({
  ^bb0(%arg0: index):
    %38 = "memref.subview"(%27, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %39 = "memref.subview"(%31, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %40 = "memref.subview"(%34, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%38, %39, %40) ({
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      "linalg.generic"(%arg1, %arg2, %arg3) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):
        %41 = "arith.addf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
        "linalg.yield"(%41) : (f64) -> ()
      }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %37 = "dma.start_transfer"(%34, %24) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
  "dma.wait_for_transfer"(%37) : (!dma.token) -> ()
  "func.return"() : () -> ()
}) {translation_info = #iree_codegen.translation_info<None>} : () -> ()
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:112:0: warning: Failed to translate kernel with xDSL
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:112:0: note: see current operation: 
quidditch_snitch.memref.microkernel(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) : memref<1x21xf64, strided<[168, 1], offset: ?>>, memref<1x21xf64, strided<[168, 1], offset: ?>>, memref<1x21xf64, strided<[168, 1], offset: ?>> {
^bb0(%arg0: memref<1x21xf64, strided<[168, 1], offset: ?>>, %arg1: memref<1x21xf64, strided<[168, 1], offset: ?>>, %arg2: memref<1x21xf64, strided<[168, 1], offset: ?>>):
  %cst = arith.constant 1.000000e+00 : f64
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x21xf64, strided<[168, 1], offset: ?>>, memref<1x21xf64, strided<[168, 1], offset: ?>>) outs(%arg2 : memref<1x21xf64, strided<[168, 1], offset: ?>>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %0 = arith.addf %in, %in_0 : f64
    %1 = arith.negf %0 : f64
    %2 = math.exp %1 : f64
    %3 = arith.addf %2, %cst : f64
    %4 = arith.divf %cst, %3 : f64
    linalg.yield %4 : f64
  }
}
