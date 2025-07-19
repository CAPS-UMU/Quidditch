<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: error: 'memref.alloca' op L1 allocations with dynamic size is currently unsupported
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: %40 = "memref.alloca"(%38) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<1x?xf64, #quidditch_snitch.l1_encoding>
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: error: 'memref.alloca' op L1 allocations with dynamic size is currently unsupported
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: %45 = "memref.alloca"(%38) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<25xxf64, #quidditch_snitch.l1_encoding>
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: error: failed to run translation of source executable to target executable for backend #hal.executable.target<"quidditch", "static", {compute_cores = 8 : i32, data_layout = "e-m:e-p:32:32-i64:64-n32-S128", target_triple = "riscv32-unknown-elf"}>
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
"hal.executable.variant"() ({
  "hal.executable.export"() ({
  ^bb0(%arg20: !hal.device):
    %66 = "arith.constant"() <{value = 1 : index}> : () -> index
    "hal.return"(%66, %66, %66) : (index, index, index) -> ()
  }) {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>], layout = #hal.pipeline.layout<push_constants = 5, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>, ordinal = 0 : index, sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"} : () -> ()
  "builtin.module"() ({
    "func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
      %0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>
      %1 = "arith.constant"() <{value = 32 : index}> : () -> index
      %2 = "arith.constant"() <{value = 25 : index}> : () -> index
      %3 = "arith.constant"() <{value = 1200 : index}> : () -> index
      %4 = "arith.constant"() <{value = 240 : index}> : () -> index
      %5 = "arith.constant"() <{value = 400 : index}> : () -> index
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
      %28 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64>
      %29 = "quidditch_snitch.compute_core_index"() : () -> index
      %30 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 150)>}> : (index) -> index
      "scf.for"(%30, %3, %3) ({
      ^bb0(%arg16: index):
        %64 = "memref.subview"(%27, %arg16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        "quidditch_snitch.memref.microkernel"(%64) ({
        ^bb0(%arg17: memref<1x150xf64, strided<[1200, 1], offset: ?>>):
          %65 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
          "linalg.fill"(%65, %arg17) <{operandSegmentSizes = array<i32: 1, 1>}> ({
          ^bb0(%arg18: f64, %arg19: f64):
            "linalg.yield"(%arg18) : (f64) -> ()
          }) : (f64, memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%6, %5, %4) ({
      ^bb0(%arg7: index):
        %48 = "affine.min"(%arg7) <{map = affine_map<(d0) -> (-d0 + 400, 240)>}> : (index) -> index
        %49 = "memref.subview"(%21, %arg7, %48) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<1x?xf64, strided<[400, 1], offset: ?>>
        %50 = "memref.alloca"(%48) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<1x?xf64>
        %51 = "memref.subview"(%50, %48) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x?xf64>, index) -> memref<1x?xf64, strided<[?, 1]>>
        %52 = "dma.start_transfer"(%49, %51) : (memref<1x?xf64, strided<[400, 1], offset: ?>>, memref<1x?xf64, strided<[?, 1]>>) -> !dma.token
        "dma.wait_for_transfer"(%52) : (!dma.token) -> ()
        "scf.for"(%6, %3, %2) ({
        ^bb0(%arg8: index):
          %53 = "memref.subview"(%22, %arg8, %arg7, %48) <{operandSegmentSizes = array<i32: 1, 2, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index, index) -> memref<25x?xf64, strided<[400, 1], offset: ?>>
          %54 = "memref.subview"(%27, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 25>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x25xf64, strided<[1200, 1], offset: ?>>
          %55 = "memref.alloca"(%48) <{alignment = 64 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<25x?xf64>
          %56 = "memref.subview"(%55, %48) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 25, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<25x?xf64>, index) -> memref<25x?xf64, strided<[?, 1]>>
          %57 = "dma.start_transfer"(%53, %56) : (memref<25x?xf64, strided<[400, 1], offset: ?>>, memref<25x?xf64, strided<[?, 1]>>) -> !dma.token
          "dma.wait_for_transfer"(%57) : (!dma.token) -> ()
          %58 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 4)>}> : (index) -> index
          "scf.for"(%58, %2, %1) ({
          ^bb0(%arg9: index):
            %59 = "affine.min"(%arg9) <{map = affine_map<(d0) -> (-d0 + 25, 4)>}> : (index) -> index
            %60 = "memref.subview"(%55, %arg9, %59, %48) <{operandSegmentSizes = array<i32: 1, 1, 2, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<25x?xf64>, index, index, index) -> memref<?x?xf64, strided<[?, 1], offset: ?>>
            %61 = "memref.subview"(%54, %arg9, %59) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<1x25xf64, strided<[1200, 1], offset: ?>>, index, index) -> memref<1x?xf64, strided<[1200, 1], offset: ?>>
            "quidditch_snitch.memref.microkernel"(%51, %60, %61) ({
            ^bb0(%arg10: memref<1x?xf64, strided<[?, 1]>>, %arg11: memref<?x?xf64, strided<[?, 1], offset: ?>>, %arg12: memref<1x?xf64, strided<[1200, 1], offset: ?>>):
              "linalg.matmul_transpose_b"(%arg10, %arg11, %arg12) <{operandSegmentSizes = array<i32: 2, 1>}> ({
              ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
                %62 = "arith.mulf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
                %63 = "arith.addf"(%arg15, %62) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
                "linalg.yield"(%63) : (f64) -> ()
              }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (memref<1x?xf64, strided<[?, 1]>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<1x?xf64, strided<[1200, 1], offset: ?>>) -> ()
            }) : (memref<1x?xf64, strided<[?, 1]>>, memref<?x?xf64, strided<[?, 1], offset: ?>>, memref<1x?xf64, strided<[1200, 1], offset: ?>>) -> ()
            "quidditch_snitch.microkernel_fence"() : () -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %31 = "arith.constant"() <{value = 9600 : index}> : () -> index
      %32 = "memref.view"(%0, %31) : (memref<100000xi8>, index) -> memref<1200xf64>
      %33 = "memref.reinterpret_cast"(%32) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
      %34 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64>
      %35 = "memref.cast"(%33) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>>
      %36 = "dma.start_transfer"(%23, %35) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>>) -> !dma.token
      "dma.wait_for_transfer"(%36) : (!dma.token) -> ()
      %37 = "arith.constant"() <{value = 19200 : index}> : () -> index
      %38 = "memref.view"(%0, %37) : (memref<100000xi8>, index) -> memref<1200xf64>
      %39 = "memref.reinterpret_cast"(%38) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
      %40 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1200xf64>
      %41 = "memref.cast"(%39) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>>
      %42 = "dma.start_transfer"(%24, %41) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>>) -> !dma.token
      "dma.wait_for_transfer"(%42) : (!dma.token) -> ()
      "scf.for"(%30, %3, %3) ({
      ^bb0(%arg0: index):
        %44 = "memref.subview"(%27, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        %45 = "memref.subview"(%33, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        %46 = "memref.subview"(%39, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>>
        "quidditch_snitch.memref.microkernel"(%44, %45, %46) ({
        ^bb0(%arg1: memref<1x150xf64, strided<[1200, 1], offset: ?>>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>>):
          "linalg.generic"(%arg1, %arg2, %arg3) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):
            %47 = "arith.addf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%47) : (f64) -> ()
          }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>, memref<1x150xf64, strided<[1200, 1], offset: ?>>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %43 = "dma.start_transfer"(%39, %24) : (memref<1x1200xf64>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
      "dma.wait_for_transfer"(%43) : (!dma.token) -> ()
      "func.return"() : () -> ()
    }) {translation_info = #iree_codegen.translation_info<None>} : () -> ()
  }) : () -> ()
  "hal.executable.variant_end"() : () -> ()
}) {sym_name = "static", target = #hal.executable.target<"quidditch", "static", {compute_cores = 8 : i32, data_layout = "e-m:e-p:32:32-i64:64-n32-S128", target_triple = "riscv32-unknown-elf"}>} : () -> ()
failed to translate executables
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:47:0: warning: RADDISH THE WHOLE FUNCTION IS HERE!
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:47:0: note: see current operation: 
func.func @main$async_dispatch_3_elementwise_400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 1.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = hal.interface.constant.load[5] : i32
  %6 = hal.interface.constant.load[6] : i32
  %7 = hal.interface.constant.load[7] : i32
  %8 = hal.interface.constant.load[8] : i32
  %9 = arith.extui %0 : i32 to i64
  %10 = arith.extui %1 : i32 to i64
  %11 = arith.shli %10, %c32_i64 : i64
  %12 = arith.ori %9, %11 : i64
  %13 = arith.index_castui %12 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %14 = arith.index_castui %2 : i32 to index
  %15 = arith.index_castui %3 : i32 to index
  %16 = arith.index_castui %4 : i32 to index
  %17 = arith.index_castui %5 : i32 to index
  %18 = arith.index_castui %6 : i32 to index
  %19 = arith.index_castui %7 : i32 to index
  %20 = arith.index_castui %8 : i32 to index
  %21 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%13) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %22 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%14) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %23 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%15) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %24 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%16) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %25 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%17) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %26 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%18) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %27 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%19) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400xf64>>
  %28 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%20) : !flow.dispatch.tensor<writeonly:tensor<400xf64>>
  %29 = flow.dispatch.tensor.load %28, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<400xf64>> -> tensor<400xf64>
  %30 = flow.dispatch.tensor.load %21, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %31 = flow.dispatch.tensor.load %22, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %32 = flow.dispatch.tensor.load %23, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %33 = flow.dispatch.tensor.load %24, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %34 = flow.dispatch.tensor.load %25, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %35 = flow.dispatch.tensor.load %26, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %36 = flow.dispatch.tensor.load %27, offsets = [0], sizes = [400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<400xf64>> -> tensor<400xf64>
  %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%30, %31, %32, %33, %34, %35, %36 : tensor<400xf64>, tensor<400xf64>, tensor<400xf64>, tensor<400xf64>, tensor<400xf64>, tensor<400xf64>, tensor<400xf64>) outs(%29 : tensor<400xf64>) {
  ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
    %38 = arith.addf %in_4, %in_5 : f64
    %39 = arith.addf %in_2, %in_3 : f64
    %40 = arith.negf %39 : f64
    %41 = math.exp %40 : f64
    %42 = arith.addf %41, %cst : f64
    %43 = arith.divf %cst, %42 : f64
    %44 = arith.mulf %in_1, %43 : f64
    %45 = arith.addf %in_0, %44 : f64
    %46 = math.tanh %45 : f64
    %47 = arith.negf %38 : f64
    %48 = math.exp %47 : f64
    %49 = arith.addf %48, %cst : f64
    %50 = arith.divf %cst, %49 : f64
    %51 = arith.subf %in, %46 : f64
    %52 = arith.mulf %51, %50 : f64
    %53 = arith.addf %52, %46 : f64
    linalg.yield %53 : f64
  } -> tensor<400xf64>
  flow.dispatch.tensor.store %37, %28, offsets = [0], sizes = [400], strides = [1] : tensor<400xf64> -> !flow.dispatch.tensor<writeonly:tensor<400xf64>>
  return
}
