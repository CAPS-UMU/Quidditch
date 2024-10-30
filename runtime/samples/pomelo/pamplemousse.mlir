builtin.module @pamplemousse {
    func.func @add(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
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
      func.return %out : tensor<4xf64>
    }
        func.func @matmulTiny(%lhs: tensor<2x3xf64>, %rhs: tensor<3x2xf64>, %acc: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<2x3xf64>, tensor<3x2xf64>)
    outs(%acc: tensor<2x2xf64>)
  -> tensor<2x2xf64>
  return %result: tensor<2x2xf64>
  // return %acc: tensor<2x2xf64>
}
}
