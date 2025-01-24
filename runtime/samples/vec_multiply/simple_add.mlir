builtin.module @test_simple_add {
    func.func @add(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
      %four = arith.constant 4 : i32
      func.call @record_cycles(%four) : (i32) -> ()

      %init = tensor.empty() : tensor<4xf64>
      %out = linalg.generic
              {indexing_maps = [affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>],
               iterator_types = ["parallel"]}
               ins(%arg0, %arg1 : tensor<4xf64>, tensor<4xf64>)
               outs(%init : tensor<4xf64>) {
      ^bb0(%in: f64 , %in_1: f64, %out: f64):
        %o = arith.addf %in, %in_1 : f64
        linalg.yield %o : f64
      } -> tensor<4xf64>
      
      func.call @record_cycles(%four) : (i32) -> ()
      func.return %out : tensor<4xf64>
    }

    // this dummy record_cycles compiles fine and seems to get called when I run test_simple_add.
    // This dummy function doesn't change the output of simple_add, 
    // so it's possible the compiler removed it and that's the only reason
    // I don't get a `not registered on the context` error when running.
    func.func @record_cycles(%arg0: i32) -> () {
      // %num = arith.constant 87 : i32
      func.return
    }

  // this function, which is the one I really want to be able to call from test_simple_add, 
  // does not seem to be added to the IREE context. 
  // I get `not registered on the context` error.
  // Is this because the actual function definition is defined in C?
  //     "func.func"() <{function_type = (i32) 
  // -> (), sym_name = "record_cycles", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()
}
