<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Let's look at ALL the allocOps before doging ANYTHING! 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1248 

allocOp with memref shape 1 40 

allocOp with memref shape 96 40 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1200 
Well,  those were all the allocOps... =_=

allocOp with memref shape 1 1200 
offset aligned to alignment=64 which is 0
memref size is 8
allocElements is 1200
NOW memref size is 9600
offset is 9600
 apparently the kernel does indeed fit in L1
shi-tzu:
allocOp with memref shape 1 1248 
offset aligned to alignment=64 which is 9600
memref size is 8
allocElements is 1248
NOW memref size is 9984
offset is 19584
 apparently the kernel does indeed fit in L1
shi-tzu:
allocOp with memref shape 1 40 
offset aligned to alignment=64 which is 19584
memref size is 8
allocElements is 40
NOW memref size is 320
offset is 19904
 apparently the kernel does indeed fit in L1
shi-tzu:
allocOp with memref shape 96 40 
offset aligned to alignment=64 which is 19904
memref size is 8
allocElements is 3840
NOW memref size is 30720
offset is 50624
 apparently the kernel does indeed fit in L1
shi-tzu:
allocOp with memref shape 1 1200 
offset aligned to alignment=64 which is 50624
memref size is 8
allocElements is 1200
NOW memref size is 9600
offset is 60224
 apparently the kernel does indeed fit in L1
shi-tzu:
allocOp with memref shape 1 1200 
offset aligned to alignment=64 which is 60224
memref size is 8
allocElements is 1200
NOW memref size is 9600
offset is 69824
 apparently the kernel does indeed fit in L1
shi-tzu:
/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
"func.func"() <{function_type = () -> (), sym_name = "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"}> ({
  %0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>
  %1 = "arith.constant"() <{value = 1200 : index}> : () -> index
  %2 = "arith.constant"() <{value = 96 : index}> : () -> index
  %3 = "arith.constant"() <{value = 1248 : index}> : () -> index
  %4 = "arith.constant"() <{value = 40 : index}> : () -> index
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
  
  // input vector to matmul_transpose_b
  %21 = "hal.interface.binding.subspan"(%17) {alignment = 64 : index, binding = 0 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} 
  : (index) -> memref<1x400xf64, strided<[400, 1], offset: ?>>
  "memref.assume_alignment"(%21) <{alignment = 64 : i32}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>) -> ()
  
  // input weight matrix to matmul_tranpose_b
  %22 = "hal.interface.binding.subspan"(%18) {alignment = 64 : index, binding = 1 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} 
  : (index) -> memref<1200x400xf64, strided<[400, 1], offset: ?>>
  "memref.assume_alignment"(%22) <{alignment = 1 : i32}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>) -> ()
  
  // input vector to fused addition
  %23 = "hal.interface.binding.subspan"(%19) {alignment = 64 : index, binding = 1 : index, descriptor_flags = 1 : i32, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} 
  : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  "memref.assume_alignment"(%23) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
  
 // output vector of fused addition / entire fused kernel
  %24 = "hal.interface.binding.subspan"(%20) {alignment = 64 : index, binding = 2 : index, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  "memref.assume_alignment"(%24) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
  
  %25 = "arith.constant"() <{value = 0 : index}> : () -> index
  %26 = "memref.view"(%0, %25) : (memref<100000xi8>, index) -> memref<1200xf64>
  %27 = "memref.reinterpret_cast"(%26) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> 
  : (memref<1200xf64>) -> memref<1x1200xf64>
  %28 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> 
  : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  
  // set 1x1200 output vector to all zeroes
  %29 = "quidditch_snitch.compute_core_index"() : () -> index
  %30 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 150)>}> : (index) -> index
  "scf.for"(%30, %1, %1) ({
  ^bb0(%arg16: index):
    %76 = "memref.subview"(%27, %arg16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> 
    : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%76) ({
    ^bb0(%arg17: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      %77 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
      "linalg.fill"(%77, %arg17) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg18: f64, %arg19: f64):
        "linalg.yield"(%arg18) : (f64) -> ()
      }) : (f64, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()


  %31 = "arith.constant"() <{value = 9600 : index}> : () -> index
  %32 = "memref.view"(%0, %31) : (memref<100000xi8>, index) -> memref<1248xf64>
  %33 = "memref.reinterpret_cast"(%32) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1248>, static_strides = array<i64: 1248, 1>}> 
  : (memref<1248xf64>) -> memref<1x1248xf64>
  
  // allocate 1x1248 in L1 
  %34 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> 
  : () -> memref<1x1248xf64, #quidditch_snitch.l1_encoding>

  // take a 1x1200 subview of that 1x1248 buffer
  %35 = "memref.subview"(%33) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1, 1>}> 
  : (memref<1x1248xf64>) 
  -> memref<1x1200xf64, strided<[1248, 1]>, #quidditch_snitch.l1_encoding>
  
  // copy the zeroed out 1200 buffer into a size 1200 subview of the 1248 size buffer, all in L1
  %36 = "dma.start_transfer"(%27, %35) : (memref<1x1200xf64>, memref<1x1200xf64, strided<[1248, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%36) : (!dma.token) -> ()

  "scf.for"(%6, %5, %4) ({ // reduction dimension slices
  ^bb0(%arg7: index):
  // select a 1x40 tile in L3
    %54 = "memref.subview"(%21, %arg7) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 40>, static_strides = array<i64: 1, 1>}> 
    : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index) -> memref<1x40xf64, strided<[400, 1], offset: ?>>
    %55 = "arith.constant"() <{value = 19584 : index}> : () -> index
    %56 = "memref.view"(%0, %55) : (memref<100000xi8>, index) -> memref<40xf64>
    %57 = "memref.reinterpret_cast"(%56) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 40>, static_strides = array<i64: 40, 1>}> 
    : (memref<40xf64>) -> memref<1x40xf64>
    %58 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x40xf64, #quidditch_snitch.l1_encoding>
    %59 = "memref.cast"(%57) : (memref<1x40xf64>) -> memref<1x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>
    // copy 1x40 L3 slice into L1
    %60 = "dma.start_transfer"(%54, %59) : (memref<1x40xf64, strided<[400, 1], offset: ?>>, memref<1x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
    "dma.wait_for_transfer"(%60) : (!dma.token) -> ()
    // 0 , 1248, 96
    "scf.for"(%6, %3, %2) ({ // row-dim slices
    ^bb0(%arg8: index):
      %61 = "affine.min"(%arg8) <{map = affine_map<(d0) -> (1200, d0 + 96)>}> : (index) -> index
      %62 = "affine.apply"(%61, %arg8) <{map = affine_map<(d0, d1) -> (d0 - d1)>}> : (index, index) -> index
      // select a weight matrix slice in L3, either of size 96x40, OR of size 48x40
      %63 = "memref.subview"(%22, %arg8, %arg7, %62) <{operandSegmentSizes = array<i32: 1, 2, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, 40>, static_strides = array<i64: 1, 1>}> 
      : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index, index) 
      -> memref<?x40xf64, strided<[400, 1], offset: ?>>
      %64 = "arith.constant"() <{value = 19904 : index}> : () -> index
      %65 = "memref.view"(%0, %64) : (memref<100000xi8>, index) -> memref<3840xf64>
      %66 = "memref.reinterpret_cast"(%65) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 96, 40>, static_strides = array<i64: 40, 1>}> 
      : (memref<3840xf64>) -> memref<96x40xf64>
      
      %67 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> 
      : () -> memref<96x40xf64, #quidditch_snitch.l1_encoding>
      
      // select a matching subview in L1, either of size 96x40 OR of size 48*40
      %68 = "memref.subview"(%66, %62) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, 40>, static_strides = array<i64: 1, 1>}> 
      : (memref<96x40xf64>, index) -> memref<?x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>
      
      // copy the weight matrix slice from L3 into L1
      %69 = "dma.start_transfer"(%63, %68) : (memref<?x40xf64, strided<[400, 1], offset: ?>>, memref<?x40xf64, strided<[40, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
      "dma.wait_for_transfer"(%69) : (!dma.token) -> ()

      // select a 1x96 slice from the output vector allocated in L1
      %70 = "memref.subview"(%33, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 96>, static_strides = array<i64: 1, 1>}> 
      : (memref<1x1248xf64>, index) 
      -> memref<1x96xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>
      
      %71 = "affine.apply"(%29) <{map = affine_map<()[s0] -> (s0 * 12)>}> : (index) -> index
      "scf.for"(%71, %2, %2) ({
      ^bb0(%arg9: index):
        %72 = "memref.subview"(%66, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 12, 40>, static_strides = array<i64: 1, 1>}> 
        : (memref<96x40xf64>, index) 
        -> memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        %73 = "memref.subview"(%70, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 12>, static_strides = array<i64: 1, 1>}> 
        : (memref<1x96xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, index) 
        -> memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>
        // take in a 1x40 slice of input in L1, and 12x40 subview of 96x40 slice in L1, and 1x12 subview of 1x96 slice.
        "quidditch_snitch.memref.microkernel"(%57, %72, %73) ({
        ^bb0(%arg10: memref<1x40xf64, #quidditch_snitch.l1_encoding>, 
             %arg11: memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>, 
             %arg12: memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          "linalg.matmul_transpose_b"(%arg10, %arg11, %arg12) <{operandSegmentSizes = array<i32: 2, 1>}> ({
          ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
            %74 = "arith.mulf"(%arg13, %arg14) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            %75 = "arith.addf"(%arg15, %74) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
            "linalg.yield"(%75) : (f64) -> ()
          }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 96, 40], l1_tiles_interchange = [2, 0, 1], myrtleCost = [1382400, 12]>} : (memref<1x40xf64, #quidditch_snitch.l1_encoding>, memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        }) : (memref<1x40xf64>, memref<12x40xf64, strided<[40, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x12xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
        "quidditch_snitch.microkernel_fence"() : () -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()


  %37 = "arith.constant"() <{value = 50624 : index}> : () -> index
  %38 = "memref.view"(%0, %37) : (memref<100000xi8>, index) -> memref<1200xf64>
  %39 = "memref.reinterpret_cast"(%38) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> 
  : (memref<1200xf64>) -> memref<1x1200xf64>
  // allocate 1x1200 in L1
  %40 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> 
  : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %41 = "memref.cast"(%39) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>  
  // copy input of fused addition from L3 to L1
  %42 = "dma.start_transfer"(%23, %41) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%42) : (!dma.token) -> ()

  %43 = "arith.constant"() <{value = 60224 : index}> : () -> index
  %44 = "memref.view"(%0, %43) : (memref<100000xi8>, index) -> memref<1200xf64>
  %45 = "memref.reinterpret_cast"(%44) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>
  // allocate 1x1200 in L1
  %46 = "memref.alloca"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> 
  : () -> memref<1x1200xf64, #quidditch_snitch.l1_encoding>
  %47 = "memref.cast"(%45) : (memref<1x1200xf64>) -> memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>  
  // copy what I think is the output vector of the entire kernel from L3 to L1?????
  // but shouldn't the values in L3 be garbage??
  %48 = "dma.start_transfer"(%24, %47) : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>, memref<1x1200xf64, strided<[1200, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
  "dma.wait_for_transfer"(%48) : (!dma.token) -> ()

  // perform fused addition on output of matmul_tranpose_b
  "scf.for"(%30, %1, %1) ({
  ^bb0(%arg0: index):
    // I think this argument is the output of the matmul_transpose_b, sized 1x1248
    %50 = "memref.subview"(%35, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> 
    : (memref<1x1200xf64, strided<[1248, 1]>, #quidditch_snitch.l1_encoding>, index) -> memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    // I *think* this argument is the other input to the fused addition
    %51 = "memref.subview"(%39, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> 
    : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    // then I *think* this must be the output vector for the fused addition
    %52 = "memref.subview"(%45, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 150>, static_strides = array<i64: 1, 1>}> 
    : (memref<1x1200xf64>, index) -> memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    "quidditch_snitch.memref.microkernel"(%50, %51, %52) ({
    ^bb0(%arg1: memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg2: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg3: memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
      "linalg.generic"(%arg1, %arg2, %arg3) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):
        %53 = "arith.addf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
        "linalg.yield"(%53) : (f64) -> ()
      }) : (memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    }) : (memref<1x150xf64, strided<[1248, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, memref<1x150xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>) -> ()
    "quidditch_snitch.microkernel_fence"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  // write the output of entire kernel back to L3
  %49 = "dma.start_transfer"(%45, %24) : (memref<1x1200xf64>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
  "dma.wait_for_transfer"(%49) : (!dma.token) -> ()
  "func.return"() : () -> ()
}) {translation_info = #iree_codegen.translation_info<None>} : () -> ()