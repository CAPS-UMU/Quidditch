<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Radish tiling level Thread This is the rewritten kernel!!!!!

/home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit.py:90:0: note: called from
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: note: see current operation: 
func.func @main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c240 = arith.constant 240 : index
  %c1200 = arith.constant 1200 : index
  %c80 = arith.constant 80 : index
  %c400 = arith.constant 400 : index
  %c0 = arith.constant 0 : index
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
  %result, %token = dma.start_tensor_copy of %21 to #quidditch_snitch.l1_encoding  -> tensor<1x1200xf64>
  %22 = dma.wait_for_tensor_copy of %21 : tensor<1x1200xf64> to %result using %token -> tensor<1x1200xf64>
  %23 = scf.forall (%arg0) = (0) to (1200) step (150) shared_outs(%arg1 = %22) -> (tensor<1x1200xf64>) {
    %extracted_slice = tensor.extract_slice %arg1[0, %arg0] [1, 150] [1, 1] : tensor<1x1200xf64> to tensor<1x150xf64>
    %29 = linalg.fill ins(%cst : f64) outs(%extracted_slice : tensor<1x150xf64>) -> tensor<1x150xf64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %29 into %arg1[0, %arg0] [1, 150] [1, 1] : tensor<1x150xf64> into tensor<1x1200xf64>
    }
  }
  %24 = scf.for %arg0 = %c0 to %c400 step %c80 iter_args(%arg1 = %23) -> (tensor<1x1200xf64>) {
    %extracted_slice = tensor.extract_slice %18[0, %arg0] [1, 80] [1, 1] : tensor<1x400xf64> to tensor<1x80xf64>
    %result_6, %token_7 = dma.start_tensor_copy of %extracted_slice to #quidditch_snitch.l1_encoding  -> tensor<1x80xf64>
    %29 = dma.wait_for_tensor_copy of %extracted_slice : tensor<1x80xf64> to %result_6 using %token_7 -> tensor<1x80xf64>
    %30 = quidditch_snitch.pipeline %c0 to %c1200 step %c240 inits(%arg1) -> tensor<1x1200xf64> {
    ^bb0(%arg2: index, %arg3: tensor<1x1200xf64>):
      %extracted_slice_8 = tensor.extract_slice %19[%arg2, %arg0] [240, 80] [1, 1] : tensor<1200x400xf64> to tensor<240x80xf64>
      %result_9, %token_10 = dma.start_tensor_copy of %extracted_slice_8 to #quidditch_snitch.l1_encoding  -> tensor<240x80xf64>
      %extracted_slice_11 = tensor.extract_slice %arg3[0, %arg2] [1, 240] [1, 1] : tensor<1x1200xf64> to tensor<1x240xf64>
      %result_12, %token_13 = dma.start_tensor_copy of %extracted_slice_11 to #quidditch_snitch.l1_encoding  -> tensor<1x240xf64>
      quidditch_snitch.pipeline_yield %arg3, %extracted_slice_8, %result_9, %token_10, %extracted_slice_11, %result_12, %token_13 : tensor<1x1200xf64>, tensor<240x80xf64>, tensor<240x80xf64>, !dma.token, tensor<1x240xf64>, tensor<1x240xf64>, !dma.token
    }, {
    ^bb0(%arg2: index, %arg3: tensor<1x1200xf64>, %arg4: tensor<240x80xf64>, %arg5: tensor<240x80xf64>, %arg6: !dma.token, %arg7: tensor<1x240xf64>, %arg8: tensor<1x240xf64>, %arg9: !dma.token):
      %31 = dma.wait_for_tensor_copy of %arg4 : tensor<240x80xf64> to %arg5 using %arg6 -> tensor<240x80xf64>
      %32 = dma.wait_for_tensor_copy of %arg7 : tensor<1x240xf64> to %arg8 using %arg9 -> tensor<1x240xf64>
      %33 = scf.forall (%arg10) = (0) to (240) step (30) shared_outs(%arg11 = %32) -> (tensor<1x240xf64>) {
        %extracted_slice_8 = tensor.extract_slice %29[0, 0] [1, 80] [1, 1] : tensor<1x80xf64> to tensor<1x80xf64>
        %extracted_slice_9 = tensor.extract_slice %31[%arg10, 0] [30, 80] [1, 1] : tensor<240x80xf64> to tensor<30x80xf64>
        %extracted_slice_10 = tensor.extract_slice %arg11[0, %arg10] [1, 30] [1, 1] : tensor<1x240xf64> to tensor<1x30xf64>
        %34 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 240, 80], l1_tiles_interchange = [0, 2, 1], dual_buffer = true>} ins(%extracted_slice_8, %extracted_slice_9 : tensor<1x80xf64>, tensor<30x80xf64>) outs(%extracted_slice_10 : tensor<1x30xf64>) -> tensor<1x30xf64>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %34 into %arg11[0, %arg10] [1, 30] [1, 1] : tensor<1x30xf64> into tensor<1x240xf64>
        }
      }
      %inserted_slice = tensor.insert_slice %33 into %arg3[0, %arg2] [1, 240] [1, 1] : tensor<1x240xf64> into tensor<1x1200xf64>
      quidditch_snitch.pipeline_yield %inserted_slice : tensor<1x1200xf64>
    }
    scf.yield %30 : tensor<1x1200xf64>
  }
  %result_0, %token_1 = dma.start_tensor_copy of %24 to #quidditch_snitch.l1_encoding  -> tensor<1x1200xf64>
  %25 = dma.wait_for_tensor_copy of %24 : tensor<1x1200xf64> to %result_0 using %token_1 -> tensor<1x1200xf64>
  %result_2, %token_3 = dma.start_tensor_copy of %20 to #quidditch_snitch.l1_encoding  -> tensor<1x1200xf64>
  %26 = dma.wait_for_tensor_copy of %20 : tensor<1x1200xf64> to %result_2 using %token_3 -> tensor<1x1200xf64>
  %result_4, %token_5 = dma.start_tensor_copy of %17 to #quidditch_snitch.l1_encoding  -> tensor<1x1200xf64>
  %27 = dma.wait_for_tensor_copy of %17 : tensor<1x1200xf64> to %result_4 using %token_5 -> tensor<1x1200xf64>
  %28 = scf.forall (%arg0) = (0) to (1200) step (150) shared_outs(%arg1 = %27) -> (tensor<1x1200xf64>) {
    %extracted_slice = tensor.extract_slice %25[0, %arg0] [1, 150] [1, 1] : tensor<1x1200xf64> to tensor<1x150xf64>
    %extracted_slice_6 = tensor.extract_slice %26[0, %arg0] [1, 150] [1, 1] : tensor<1x1200xf64> to tensor<1x150xf64>
    %extracted_slice_7 = tensor.extract_slice %arg1[0, %arg0] [1, 150] [1, 1] : tensor<1x1200xf64> to tensor<1x150xf64>
    %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_6 : tensor<1x150xf64>, tensor<1x150xf64>) outs(%extracted_slice_7 : tensor<1x150xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64):
      %30 = arith.addf %in, %in_8 : f64
      linalg.yield %30 : f64
    } -> tensor<1x150xf64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %29 into %arg1[0, %arg0] [1, 150] [1, 1] : tensor<1x150xf64> into tensor<1x1200xf64>
    }
  }
  flow.dispatch.tensor.store %28, %16, offsets = [0, 0], sizes = [1, 1200], strides = [1, 1] : tensor<1x1200xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x1200xf64>>
  return
}