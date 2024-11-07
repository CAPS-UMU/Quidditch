<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Let's look at ALL the allocOps before doging ANYTHING! 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 80 

allocOp with memref shape 240 80 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1200 
Well, those were all the allocOps... =_=

allocOp with memref shape 1 1200 
offset is 0

allocOp with memref shape 1 80 
offset is 9600

allocOp with memref shape 240 80 
offset is 10240

allocElements is 19200
memref size is 153600
offset is 163840
l1MemoryBytes is 100000, so 63840 too much
kernel does not fit into L1 memory and cannot be compiled
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
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
  ^bb0(%arg16: index):
    %61 = "memref.subview"(%26, %arg16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%61) ({
    ^bb0(%arg17: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %62 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      "linalg.fill"(%62, %arg17) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg18: f64, %arg19: f64):
        "linalg.yield"(%arg18) : (f64) -> ()
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
    "scf.for"(%5, %2, %1) ({
    ^bb0(%arg8: index):
      %48 = "memref.subview"(%21, %arg8, %arg7) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<240x80xf64, strided<[400, 1], offset: ?>>
      %49 = "memref.subview"(%26, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 240>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x240xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      %50 = "arith.constant"() <{value = 10240 : index}> : () -> index
      %51 = "memref.view"(%0, %50) : (memref<100000xi8>, index) -> memref<19200xf64>
      %52 = "memref.reinterpret_cast"(%51) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 80, 1>}> : (memref<19200xf64>) -> memref<240x80xf64>
      %53 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<240x80xf64, #quidditch_snitch.l1_encoding>
      %54 = "memref.cast"(%52) : (memref<240x80xf64>) -> memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>
      %55 = "dma.start_transfer"(%48, %54) : (memref<240x80xf64, strided<[400, 1], offset: ?>>, memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
      "dma.wait_for_transfer"(%55) : (!dma.token) -> ()
      %56 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 30)>}> : (index) -> index
      "scf.for"(%56, %1, %1) ({
      ^bb0(%arg9: index):
        %57 = "memref.subview"(%52, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 30, 80>, static_strides = array<i64: 1, 1>}> : (memref<240x80xf64>, index) -> memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %58 = "memref.subview"(%49, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 30>, static_strides = array<i64: 1, 1>}> : (memref<1x240xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, index) -> memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        "quidditch_snitch.memref.microkernel"(%44, %57, %58) ({
        ^bb0(%arg10: memref<1x80xf64, #quidditch_snitch.l1_encoding>, %arg11: memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg12: memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          "linalg.matmul_transpose_b"(%arg10, %arg11, %arg12) <{operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
            %59 = "arith.mulf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            %60 = "arith.addf"(%arg15, %59) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%60) : (f64) -> ()
          }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 240, 80], l1_tiles_interchange = [0, 2, 1]>} : (memref<1x80xf64, #quidditch_snitch.l1_encoding>, memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        }) : (memref<1x80xf64>, memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
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