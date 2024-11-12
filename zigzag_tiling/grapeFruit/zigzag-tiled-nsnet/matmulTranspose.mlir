func.func @matmultranspose(%lhs: tensor<1x400xf64>, %rhs: tensor<1200x400xf64>, %acc: tensor<1x1200xf64>) -> tensor<1x1200xf64> {
  %24 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<1x400xf64>, tensor<1200x400xf64>) outs(%acc : tensor<1x1200xf64>) -> tensor<1x1200xf64> 
 return %24 : tensor<1x1200xf64> 
}