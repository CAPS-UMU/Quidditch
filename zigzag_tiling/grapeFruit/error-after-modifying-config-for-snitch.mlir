<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: kernel does not fit into L1 memory and cannot be compiled
/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
"func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
  %0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>
  %1 = "arith.constant"() <{value = 240 : index}> : () -> index
  %2 = "arith.constant"() <{value = 1200 : index}> : () -> index
  %3 = "arith.constant"() <{value = 80 : index}> : () -> index
  %4 = "arith.constant"() <{value = 400 : index}> : () -> index
  %5 = "arith.constant"() <{value = 0 : index}> : () -> index
  %6 = "arith.constant"() <{value = 32 : i64}> : () -> i64
  %7 = "hal.interface.constant.load"() {index = 0 : index} : () -> i32
  %8 = "hal.interface.constant.load"() {index = 1 : index} : () -> i32
  %9 = "hal.interface.constant.load"() {index = 2 : index} : () -> i32
  %10 = "hal.interface.constant.load"() {index = 3 : index} : () -> i32
  %11 = "hal.interface.constant.load"() {index = 4 : index} : () -> i32
  %12 = "arith.extui"(%7) : (i32) -> i64
  %13 = "arith.extui"(%8) : (i32) -> i64
  %14 = "arith.shli"(%13, %6) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
  %15 = "arith.ori"(%12, %14) : (i64, i64) -> i64
  %16 = "arith.index_castui"(%15) {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : (i64) -> index
  %17 = "arith.index_castui"(%9) : (i32) -> index
  %18 = "arith.index_castui"(%10) : (i32) -> index
  %19 = "arith.index_castui"(%11) : (i32) -> index
  %20 = "hal.interface.binding.subspan"(%16) {alignment = 64 : index, binding = 0 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x400xf64, strided<[400, 1], offset: ?>>
  "memref.assume_alignment"(%20) <{alignment = 64 : i32}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>) -> ()
  %21 = "hal.interface.binding.subspan"(%17) {alignment = 64 : index, binding = 1 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1200x400xf64, strided<[400, 1], offset: ?>>
  "memref.assume_alignment"(%21) <{alignment = 1 : i32}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>) -> ()
  %22 = "hal.interface.binding.subspan"(%18) {alignment = 64 : index, binding = 1 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  "memref.assume_alignment"(%22) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
  %23 = "hal.interface.binding.subspan"(%19) {alignment = 64 : index, binding = 2 : index, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  "memref.assume_alignment"(%23) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
  %24 = "arith.constant"() <{value = 0 : index}> : () -> index
  %25 = "memref.view"(%0, %24) : (memref<100000xi8>, index) -> memref<1200xf64>
  %26 = "memref.reinterpret_cast"(%25) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
  %27 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %28 = "quidditch_snitch.compute_core_index"() : () -> index
  %29 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 150)>}> : (index) -> index
  "scf.for"(%29, %2, %2) ({
  ^bb0(%arg25: index):
    %75 = "memref.subview"(%26, %arg25) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%75) ({
    ^bb0(%arg26: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %76 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      "linalg.fill"(%76, %arg26) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg27: f64, %arg28: f64):
        "linalg.yield"(%arg27) : (f64) -> ()
      }) : (f64, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%5, %4, %3) ({
  ^bb0(%arg7: index):
    %41 = "memref.subview"(%20, %arg7) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 80>, static_strides = array<i64: 1, 1>}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index) -> memref<1x80xf64, strided<[400, 1], offset: ?>>
    %42 = "arith.constant"() <{value = 9600 : index}> : () -> index
    %43 = "memref.view"(%0, %42) : (memref<100000xi8>, index) -> memref<80xf64>
    %44 = "memref.reinterpret_cast"(%43) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 80>, static_strides = array<i64: 80, 1>}> : (memref<80xf64>) -> memref<1x80xf64>
    %45 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x80xf64, #quidditch_snitch.l1_encoding>
    %46 = "memref.cast"(%44) : (memref<1x80xf64>) -> memref<1x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>
    %47 = "dma.start_transfer"(%41, %46) : (memref<1x80xf64, strided<[400, 1], offset: ?>>, memref<1x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
    "dma.wait_for_transfer"(%47) : (!dma.token) -> ()
    %48 = "arith.constant"() <{value = 10240 : index}> : () -> index
    %49 = "memref.view"(%0, %48) : (memref<100000xi8>, index) -> memref<19200xf64>
    %50 = "memref.reinterpret_cast"(%49) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 80, 1>}> : (memref<19200xf64>) -> memref<240x80xf64>
    %51 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<240x80xf64, #quidditch_snitch.l1_encoding>
    %52 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<240x80xf64, #quidditch_snitch.l1_encoding>
    %53 = "memref.subview"(%21, %arg7) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index) -> memref<240x80xf64, strided<[400, 1], offset: ?>>
    %54 = "memref.cast"(%50) : (memref<240x80xf64>) -> memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>
    %55 = "dma.start_transfer"(%53, %54) : (memref<240x80xf64, strided<[400, 1], offset: ?>>, memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
    %56:2 = "scf.for"(%1, %2, %1, %50, %55) ({
    ^bb0(%arg15: index, %arg16: memref<240x80xf64, #quidditch_snitch.l1_encoding>, %arg17: !dma.token):
      %63 = "memref.subview"(%21, %arg15, %arg7) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<240x80xf64, strided<[400, 1], offset: ?>>
      %64 = "affine.apply"(%arg15) <{map = affine_map<(d0) -> ((d0 floordiv 240) mod 2)>}> : (index) -> index
      %65 = "scf.index_switch"(%64) <{cases = array<i64: 0>}> ({
        "scf.yield"(%52) : (memref<240x80xf64, #quidditch_snitch.l1_encoding>) -> ()
      }, {
        "scf.yield"(%50) : (memref<240x80xf64>) -> ()
      }) : (index) -> memref<240x80xf64, #quidditch_snitch.l1_encoding>
      %66 = "memref.cast"(%65) : (memref<240x80xf64, #quidditch_snitch.l1_encoding>) -> memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>
      %67 = "dma.start_transfer"(%63, %66) : (memref<240x80xf64, strided<[400, 1], offset: ?>>, memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
      %68 = "affine.apply"(%arg15) <{map = affine_map<(d0) -> (d0 - 240)>}> : (index) -> index
      %69 = "memref.subview"(%26, %68) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 240>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x240xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      "dma.wait_for_transfer"(%arg17) : (!dma.token) -> ()
      %70 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 30)>}> : (index) -> index
      "scf.for"(%70, %1, %1) ({
      ^bb0(%arg18: index):
        %71 = "memref.subview"(%arg16, %arg18) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 30, 80>, static_strides = array<i64: 1, 1>}> : (memref<240x80xf64, #quidditch_snitch.l1_encoding>, index) -> memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %72 = "memref.subview"(%69, %arg18) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 30>, static_strides = array<i64: 1, 1>}> : (memref<1x240xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, index) -> memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        "quidditch_snitch.memref.microkernel"(%44, %71, %72) ({
        ^bb0(%arg19: memref<1x80xf64, #quidditch_snitch.l1_encoding>, %arg20: memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg21: memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          "linalg.matmul_transpose_b"(%arg19, %arg20, %arg21) <{operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg22: f64, %arg23: f64, %arg24: f64):
            %73 = "arith.mulf"(%arg22, %arg23) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            %74 = "arith.addf"(%arg24, %73) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%74) : (f64) -> ()
          }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 240, 80], l1_tiles_interchange = [0, 2, 1], dual_buffer = true>} : (memref<1x80xf64, #quidditch_snitch.l1_encoding>, memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        }) : (memref<1x80xf64>, memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"(%65, %67) : (memref<240x80xf64, #quidditch_snitch.l1_encoding>, !dma.token) -> ()
    }) : (index, index, index, memref<240x80xf64>, !dma.token) -> (memref<240x80xf64, #quidditch_snitch.l1_encoding>, !dma.token)
    %57 = "memref.subview"(%26) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 960>, static_sizes = array<i64: 1, 240>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>) -> memref<1x240xf64, strided<[1200, 1], offset: 960>, #quidditch_snitch.l1_encoding>
    "dma.wait_for_transfer"(%56#1) : (!dma.token) -> ()
    %58 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 30)>}> : (index) -> index
    "scf.for"(%58, %1, %1) ({
    ^bb0(%arg8: index):
      %59 = "memref.subview"(%56#0, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 30, 80>, static_strides = array<i64: 1, 1>}> : (memref<240x80xf64, #quidditch_snitch.l1_encoding>, index) -> memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      %60 = "memref.subview"(%57, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 30>, static_strides = array<i64: 1, 1>}> : (memref<1x240xf64, strided<[1200, 1], offset: 960>, #quidditch_snitch.l1_encoding>, index) -> memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      "quidditch_snitch.memref.microkernel"(%44, %59, %60) ({
      ^bb0(%arg9: memref<1x80xf64, #quidditch_snitch.l1_encoding>, %arg10: memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg11: memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
        "linalg.matmul_transpose_b"(%arg9, %arg10, %arg11) <{operandSegmentSizes = array<i32: 2, 1>}> ({
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
          %61 = "arith.mulf"(%arg12, %arg13) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          %62 = "arith.addf"(%arg14, %61) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          "linalg.yield"(%62) : (f64) -> ()
        }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 240, 80], l1_tiles_interchange = [0, 2, 1], dual_buffer = true>} : (memref<1x80xf64, #quidditch_snitch.l1_encoding>, memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
      }) : (memref<1x80xf64>, memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
      "quidditch_snitch.microkernel_fence"() : () -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %30 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %31 = "memref.cast"(%30) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %32 = "dma.start_transfer"(%22, %31) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%32) : (!dma.token) -> ()
  %33 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %34 = "memref.cast"(%33) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %35 = "dma.start_transfer"(%23, %34) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%35) : (!dma.token) -> ()
  "scf.for"(%29, %2, %2) ({
  ^bb0(%arg0: index):
    %37 = "memref.subview"(%26, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %38 = "memref.subview"(%30, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %39 = "memref.subview"(%33, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%37, %38, %39) ({
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      "linalg.generic"(%arg1, %arg2, %arg3) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):
        %40 = "arith.addf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
        "linalg.yield"(%40) : (f64) -> ()
      }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %36 = "dma.start_transfer"(%33, %23) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
  "dma.wait_for_transfer"(%36) : (!dma.token) -> ()
  "func.return"() : () -> ()
}) {translation_info = #iree_codegen.translation_info<None>} : () -> ()

