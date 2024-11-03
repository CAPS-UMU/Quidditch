builtin.module @pumpkin {
  // func.func @matmulTiny(%lhs: tensor<104x104xf64>, %rhs: tensor<104x104xf64>, %acc: tensor<104x104xf64>) -> tensor<104x104xf64> {
  // %result = linalg.matmul // changing linalg.add to linalg.matmul, causes a RESOURCE_EXHAUSTED error
  //   ins(%lhs, %rhs: tensor<104x104xf64>, tensor<104x104xf64>)
  //   outs(%acc: tensor<104x104xf64>)
  // -> tensor<104x104xf64>
  // return %result: tensor<104x104xf64>
  // }
    func.func @matmulTiny(%lhs: tensor<8x8xf64>, %rhs: tensor<8x8xf64>, %acc: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %result = linalg.matmul 
    ins(%lhs, %rhs: tensor<8x8xf64>, tensor<8x8xf64>)
    outs(%acc: tensor<8x8xf64>)
  -> tensor<8x8xf64>
  return %result: tensor<8x8xf64>
  }
  // func.func @matmulTiny(%lhs: tensor<3x3xf64>, %rhs: tensor<3x3xf64>, %acc: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // %result = linalg.add // changing linalg.add to linalg.matmul, causes a RESOURCE_EXHAUSTED error
  //   ins(%lhs, %rhs: tensor<3x3xf64>, tensor<3x3xf64>)
  //   outs(%acc: tensor<3x3xf64>)
  // -> tensor<3x3xf64>
  // return %result: tensor<3x3xf64>
  // }
}

// caused by linalg.matmul, but not linalg.add
// [fesvr] Wrote 36 bytes of bootrom to 0x1000
// [fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
// [fesvr] Wrote 56 bytes of bootdata to 0x1024
// [Tracer] Logging Hart          8 to logs/trace_hart_00000008.dasm
// [Tracer] Logging Hart          0 to logs/trace_hart_00000000.dasm
// [Tracer] Logging Hart          1 to logs/trace_hart_00000001.dasm
// [Tracer] Logging Hart          2 to logs/trace_hart_00000002.dasm
// [Tracer] Logging Hart          3 to logs/trace_hart_00000003.dasm
// [Tracer] Logging Hart          4 to logs/trace_hart_00000004.dasm
// [Tracer] Logging Hart          5 to logs/trace_hart_00000005.dasm
// [Tracer] Logging Hart          6 to logs/trace_hart_00000006.dasm
// [Tracer] Logging Hart          7 to logs/trace_hart_00000007.dasm
// RESOURCE_EXHAUSTED; while invoking native function pamplemousse.matmulTiny; 
// [ 1]   native hal.fence.create:0 -
// [ 0]   native pamplemousse.matmulTiny:0 -
