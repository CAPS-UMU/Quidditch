<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Radish tiling level L1 This is the rewritten kernel!!!!!

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.extui %0 : i32 to i64
  %6 = arith.extui %1 : i32 to i64
  %7 = arith.shli %6, %c32_i64 : i64
  %8 = arith.ori %5, %7 : i64
  %9 = arith.index_castui %8 {stream.alignment = 128 : index, stream.values = [0 : index, 3200 : index]} : i64 to index
  %10 = arith.index_castui %2 : i32 to index
  %11 = arith.index_castui %3 : i32 to index
  %12 = arith.index_castui %4 : i32 to index
  %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>>
  %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%11) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>>
  %16 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%12) : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  %17 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %19 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [1200, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x400xf64>> -> tensor<1200x400xf64>
  %20 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1200xf64>> -> tensor<1x1200xf64>
  %21 = tensor.empty() : tensor<1x1200xf64>
  %22 = linalg.fill ins(%cst : f64) outs(%21 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
  %c0 = arith.constant 0 : index
  %c400 = arith.constant 400 : index
  %c80 = arith.constant 80 : index
  %23 = scf.for %arg0 = %c0 to %c400 step %c80 iter_args(%arg1 = %22) -> (tensor<1x1200xf64>) {
    %c0_0 = arith.constant 0 : index
    %c1200 = arith.constant 1200 : index
    %c240 = arith.constant 240 : index
    %25 = scf.for %arg2 = %c0_0 to %c1200 step %c240 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
      %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, 80] [1, 1] : tensor<1x400xf64> to tensor<1x80xf64>
      %extracted_slice_1 = tensor.extract_slice %19[%arg2, %arg0] [240, 80] [1, 1] : tensor<1200x400xf64> to tensor<240x80xf64>
      %extracted_slice_2 = tensor.extract_slice %arg3[0, %arg2] [1, 240] [1, 1] : tensor<1x1200xf64> to tensor<1x240xf64>
      %26 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 240, 80], l1_tiles_interchange = [0, 2, 1], dual_buffer = true>} ins(%extracted_slice, %extracted_slice_1 : tensor<1x80xf64>, tensor<240x80xf64>) outs(%extracted_slice_2 : tensor<1x240xf64>) -> tensor<1x240xf64>
      %inserted_slice = tensor.insert_slice %26 into %arg3[0, %arg2] [1, 240] [1, 1] : tensor<1x240xf64> into tensor<1x1200xf64>
      scf.yield %inserted_slice : tensor<1x1200xf64>
    }
    scf.yield %25 : tensor<1x1200xf64>
  }
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %20 : tensor<1x1200xf64>, tensor<1x1200xf64>) outs(%17 : tensor<1x1200xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %25 = arith.addf %in, %in_0 : f64
    linalg.yield %25 : f64
  } -> tensor<1x1200xf64>
  flow.dispatch.tensor.store %24, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}