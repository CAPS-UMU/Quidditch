/home/hoppip/Quidditch/runtime/samples/nsnet2/NsNet2.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
"func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
  %0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>
  %1 = "arith.constant"() <{value = 40 : index}> : () -> index
  %2 = "arith.constant"() <{value = 1200 : index}> : () -> index
  %3 = "arith.constant"() <{value = 100 : index}> : () -> index
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
  ^bb0(%arg16: index):
    %67 = "memref.subview"(%26, %arg16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%67) ({
    ^bb0(%arg17: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %68 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      "linalg.fill"(%68, %arg17) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg18: f64, %arg19: f64):
        "linalg.yield"(%arg18) : (f64) -> ()
      }) : (f64, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  "scf.for"(%5, %4, %3) ({
  ^bb0(%arg7: index):
    %47 = "memref.subview"(%20, %arg7) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 100>, static_strides = array<i64: 1, 1>}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index) -> memref<1x100xf64, strided<[400, 1], offset: ?>>
    %48 = "arith.constant"() <{value = 9600 : index}> : () -> index
    %49 = "memref.view"(%0, %48) : (memref<100000xi8>, index) -> memref<100xf64>
    %50 = "memref.reinterpret_cast"(%49) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 100>, static_strides = array<i64: 100, 1>}> : (memref<100xf64>) -> memref<1x100xf64>
    %51 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x100xf64, #quidditch_snitch.l1_encoding>
    %52 = "memref.cast"(%50) : (memref<1x100xf64>) -> memref<1x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
    %53 = "dma.start_transfer"(%47, %52) : (memref<1x100xf64, strided<[400, 1], offset: ?>>, memref<1x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
    "dma.wait_for_transfer"(%53) : (!dma.token) -> ()
    "scf.for"(%5, %2, %1) ({
    ^bb0(%arg8: index):
      %54 = "memref.subview"(%21, %arg8, %arg7) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 40, 100>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<40x100xf64, strided<[400, 1], offset: ?>>
      %55 = "memref.subview"(%26, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 40>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x40xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      %56 = "arith.constant"() <{value = 10432 : index}> : () -> index
      %57 = "memref.view"(%0, %56) : (memref<100000xi8>, index) -> memref<4000xf64>
      %58 = "memref.reinterpret_cast"(%57) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 40, 100>, static_strides = array<i64: 100, 1>}> : (memref<4000xf64>) -> memref<40x100xf64>
      %59 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<40x100xf64, #quidditch_snitch.l1_encoding>
      %60 = "memref.cast"(%58) : (memref<40x100xf64>) -> memref<40x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>
      %61 = "dma.start_transfer"(%54, %60) : (memref<40x100xf64, strided<[400, 1], offset: ?>>, memref<40x100xf64, strided<[100, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
      "dma.wait_for_transfer"(%61) : (!dma.token) -> ()
      %62 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 5)>}> : (index) -> index
      "scf.for"(%62, %1, %1) ({
      ^bb0(%arg9: index):
        %63 = "memref.subview"(%58, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 5, 100>, static_strides = array<i64: 1, 1>}> : (memref<40x100xf64>, index) -> memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %64 = "memref.subview"(%55, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 5>, static_strides = array<i64: 1, 1>}> : (memref<1x40xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, index) -> memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        "quidditch_snitch.memref.microkernel"(%50, %63, %64) ({
        ^bb0(%arg10: memref<1x100xf64, #quidditch_snitch.l1_encoding>, %arg11: memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg12: memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          "linalg.matmul_transpose_b"(%arg10, %arg11, %arg12) <{operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
            %65 = "arith.mulf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            %66 = "arith.addf"(%arg15, %65) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%66) : (f64) -> ()
          }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40, 100], l1_tiles_interchange = [2, 0, 1]>} : (memref<1x100xf64, #quidditch_snitch.l1_encoding>, memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        }) : (memref<1x100xf64>, memref<5x100xf64, strided<[100, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x5xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %30 = "arith.constant"() <{value = 42432 : index}> : () -> index
  %31 = "memref.view"(%0, %30) : (memref<100000xi8>, index) -> memref<1200xf64>
  %32 = "memref.reinterpret_cast"(%31) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
  %33 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %34 = "memref.cast"(%32) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %35 = "dma.start_transfer"(%22, %34) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%35) : (!dma.token) -> ()
  %36 = "arith.constant"() <{value = 52032 : index}> : () -> index
  %37 = "memref.view"(%0, %36) : (memref<100000xi8>, index) -> memref<1200xf64>
  %38 = "memref.reinterpret_cast"(%37) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
  %39 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %40 = "memref.cast"(%38) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>
  %41 = "dma.start_transfer"(%23, %40) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%41) : (!dma.token) -> ()
  "scf.for"(%29, %2, %2) ({
  ^bb0(%arg0: index):
    %43 = "memref.subview"(%26, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %44 = "memref.subview"(%32, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    %45 = "memref.subview"(%38, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%43, %44, %45) ({
    ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      "linalg.generic"(%arg1, %arg2, %arg3) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):
        %46 = "arith.addf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
        "linalg.yield"(%46) : (f64) -> ()
      }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %42 = "dma.start_transfer"(%38, %23) : (memref<1x1200xf64>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
  "dma.wait_for_transfer"(%42) : (!dma.token) -> ()
  "func.return"() : () -> ()
}) {translation_info = #iree_codegen.translation_info<None>} : () -> ()