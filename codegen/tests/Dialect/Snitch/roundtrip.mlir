// RUN: quidditch-opt %s --verify-roundtrip

func.func @test(%arg0 : memref<f64>) {
  quidditch_snitch.xdsl_kernel(%arg0) : memref<f64> {
  ^bb0(%arg1 : memref<f64>):

  }
  return
}
