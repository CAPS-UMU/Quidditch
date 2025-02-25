#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @kernel_matmul(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32>) attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [#map, #map1, #map2, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, i32, i32) outs(%arg2 : memref<104x104xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i32, %in_2: i32, %out: i32):
      %0 = arith.extsi %in : i8 to i32
      %1 = arith.subi %0, %in_1 : i32
      %2 = arith.extsi %in_0 : i8 to i32
      %3 = arith.subi %2, %in_2 : i32
      %4 = arith.muli %1, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    }
    return
  }
  func.func private @sendWorkToAccelerator(memref<8x104xi8, strided<[104, 1], offset: ?>>, memref<104x104xi8, strided<[1, 104]>>, memref<8x104xi32, strided<[104, 1], offset: ?>>) attributes {llvm.emit_c_interface}
  func.func @tiled_matmul_w_subviews(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32, strided<[104, 1]>>) attributes {llvm.emit_c_interface} {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_2 = arith.constant 8 : index
    %c13 = arith.constant 13 : index
    scf.for %arg3 = %c0 to %c13 step %c1 {
      %subview = memref.subview %arg2[%arg3, %c0] [8, 104] [1, 1] : memref<104x104xi32, strided<[104, 1]>> to memref<8x104xi32, strided<[104, 1], offset: ?>>
      %subview_3 = memref.subview %arg0[%arg3, %c0] [8, 104] [1, 1] : memref<104x104xi8> to memref<8x104xi8, strided<[104, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c13 step %c1 {
        scf.for %arg5 = %c0 to %c13 step %c1 {
          scf.for %arg6 = %c0 to %c8_2 step %c1 {
            scf.for %arg7 = %c0 to %c8_2 step %c1 {
              scf.for %arg8 = %c0 to %c8_2 step %c1 {
                %0 = arith.muli %arg3, %c8 : index
                %1 = arith.addi %0, %arg6 : index
                %2 = arith.muli %arg4, %c8_0 : index
                %3 = arith.addi %2, %arg7 : index
                %4 = arith.muli %arg5, %c8_1 : index
                %5 = arith.addi %4, %arg8 : index
                %6 = memref.load %arg0[%1, %5] : memref<104x104xi8>
                %7 = arith.extsi %6 : i8 to i32
                %8 = memref.load %arg1[%5, %3] : memref<104x104xi8, strided<[1, 104]>>
                %9 = arith.extsi %8 : i8 to i32
                %10 = memref.load %arg2[%1, %3] : memref<104x104xi32, strided<[104, 1]>>
                %11 = arith.muli %7, %9 : i32
                %12 = arith.addi %11, %10 : i32
                memref.store %12, %arg2[%1, %3] : memref<104x104xi32, strided<[104, 1]>>
              }
            }
          }
        }
      }
    }
    return
  }
  func.func @accelerator_work(%arg0: memref<8x104xi8, strided<[104, 1], offset: ?>>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<8x104xi32, strided<[104, 1], offset: ?>>) attributes {llvm.emit_c_interface} {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_1 = arith.constant 8 : index
    %c13 = arith.constant 13 : index
    scf.for %arg3 = %c0 to %c13 step %c1 {
      scf.for %arg4 = %c0 to %c13 step %c1 {
        scf.for %arg5 = %c0 to %c8_1 step %c1 {
          scf.for %arg6 = %c0 to %c8_1 step %c1 {
            scf.for %arg7 = %c0 to %c8_1 step %c1 {
              %c0_2 = arith.constant 0 : index
              %0 = arith.addi %c0_2, %arg5 : index
              %1 = arith.muli %arg3, %c8 : index
              %2 = arith.addi %1, %arg6 : index
              %3 = arith.muli %arg4, %c8_0 : index
              %4 = arith.addi %3, %arg7 : index
              %5 = memref.load %arg0[%0, %4] : memref<8x104xi8, strided<[104, 1], offset: ?>>
              %6 = arith.extsi %5 : i8 to i32
              %7 = memref.load %arg1[%4, %2] : memref<104x104xi8, strided<[1, 104]>>
              %8 = arith.extsi %7 : i8 to i32
              %9 = memref.load %arg2[%0, %2] : memref<8x104xi32, strided<[104, 1], offset: ?>>
              %10 = arith.muli %6, %8 : i32
              %11 = arith.addi %10, %9 : i32
              memref.store %11, %arg2[%0, %2] : memref<8x104xi32, strided<[104, 1], offset: ?>>
            }
          }
        }
      }
    }
    return
  }
  func.func private @modify_output(memref<8x104xi8, strided<[104, 1], offset: ?>>, memref<104x104xi8, strided<[1, 104]>>, memref<8x104xi32, strided<[104, 1], offset: ?>>) attributes {llvm.emit_c_interface}
  func.func private @dispatch_to_accelerator(memref<8x104xi8, strided<[104, 1], offset: ?>>, memref<8x104xi8, strided<[104, 1], offset: ?>>, memref<104x104xi8, strided<[1, 104]>>, memref<8x104xi32, strided<[104, 1], offset: ?>>) attributes {llvm.emit_c_interface}
  func.func @tiled_matmul(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32, strided<[104, 1]>>, %arg3: memref<104x104xi32, strided<[104, 1]>>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c13 = arith.constant 13 : index
    %c104 = arith.constant 104 : index
    %c8_0 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg4 = %c0 to %c13 step %c1 {
      %0 = arith.muli %arg4, %c8_0 : index
      %subview = memref.subview %arg2[%0, %c0] [8, 104] [1, 1] : memref<104x104xi32, strided<[104, 1]>> to memref<8x104xi32, strided<[104, 1], offset: ?>>
      %subview_1 = memref.subview %arg3[%c0, %c0] [8, 104] [1, 1] : memref<104x104xi32, strided<[104, 1]>> to memref<8x104xi32, strided<[104, 1], offset: ?>>
      %subview_2 = memref.subview %arg0[%0, %c0] [8, 104] [1, 1] : memref<104x104xi8> to memref<8x104xi8, strided<[104, 1], offset: ?>>
      func.call @dispatch_to_accelerator(%subview_2, %subview_2, %arg1, %subview_1) : (memref<8x104xi8, strided<[104, 1], offset: ?>>, memref<8x104xi8, strided<[104, 1], offset: ?>>, memref<104x104xi8, strided<[1, 104]>>, memref<8x104xi32, strided<[104, 1], offset: ?>>) -> ()
      memref.copy %subview_1, %subview : memref<8x104xi32, strided<[104, 1], offset: ?>> to memref<8x104xi32, strided<[104, 1], offset: ?>>
      scf.for %arg5 = %c0 to %c8 step %c1 {
        scf.for %arg6 = %c0 to %c104 step %c1 {
          memref.store %c0_i32, %subview_1[%arg5, %arg6] : memref<8x104xi32, strided<[104, 1], offset: ?>>
        }
      }
    }
    return
  }
  func.func private @print_my_arg(memref<104x104xi8>) attributes {llvm.emit_c_interface}
  func.func private @print_my_arg2(!llvm.ptr) attributes {llvm.emit_c_interface}
  func.func private @print_my_arg3(i64) attributes {llvm.emit_c_interface}
  func.func @sendingMemref(%arg0: memref<104x104xi8>) attributes {llvm.emit_c_interface} {
    %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<104x104xi8> -> index
    %0 = arith.index_cast %intptr : index to i64
    %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
    call @print_my_arg(%arg0) : (memref<104x104xi8>) -> ()
    return
  }
  func.func @dummy(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32, strided<[104, 1]>>, %arg3: memref<104x104xi32, strided<[104, 1]>>) attributes {llvm.emit_c_interface} {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_2 = arith.constant 8 : index
    %c13 = arith.constant 13 : index
    scf.for %arg4 = %c0 to %c13 step %c1 {
      %subview = memref.subview %arg2[%arg4, %c0] [8, 104] [1, 1] : memref<104x104xi32, strided<[104, 1]>> to memref<8x104xi32, strided<[104, 1], offset: ?>>
      %subview_3 = memref.subview %arg0[%arg4, %c0] [8, 104] [1, 1] : memref<104x104xi8> to memref<8x104xi8, strided<[104, 1], offset: ?>>
      scf.for %arg5 = %c0 to %c13 step %c1 {
        scf.for %arg6 = %c0 to %c13 step %c1 {
          scf.for %arg7 = %c0 to %c8_2 step %c1 {
            scf.for %arg8 = %c0 to %c8_2 step %c1 {
              scf.for %arg9 = %c0 to %c8_2 step %c1 {
                %0 = arith.muli %arg4, %c8 : index
                %1 = arith.addi %0, %arg7 : index
                %2 = arith.muli %arg5, %c8_0 : index
                %3 = arith.addi %2, %arg8 : index
                %4 = arith.muli %arg6, %c8_1 : index
                %5 = arith.addi %4, %arg9 : index
                %6 = memref.load %arg0[%1, %5] : memref<104x104xi8>
                %7 = arith.extsi %6 : i8 to i32
                %8 = memref.load %arg1[%5, %3] : memref<104x104xi8, strided<[1, 104]>>
                %9 = arith.extsi %8 : i8 to i32
                %10 = memref.load %arg2[%1, %3] : memref<104x104xi32, strided<[104, 1]>>
                %11 = arith.muli %7, %9 : i32
                %12 = arith.addi %11, %10 : i32
                memref.store %12, %arg2[%1, %3] : memref<104x104xi32, strided<[104, 1]>>
              }
            }
          }
        }
      }
    }
    return
  }
  func.func private @hola(memref<2x16xi8>, memref<16x2xi8, strided<[1, 16]>>, memref<2x2xi32, strided<[16, 1]>>) attributes {llvm.emit_c_interface}
  func.func @mlirFunc(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32>) attributes {llvm.emit_c_interface} {
    %cast = memref.cast %arg2 : memref<104x104xi32> to memref<104x104xi32, strided<[104, 1]>>
    call @tiled_matmul_w_subviews(%arg0, %arg1, %cast) : (memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, memref<104x104xi32, strided<[104, 1]>>) -> ()
    return
  }
  func.func @matmul(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32>) attributes {llvm.emit_c_interface} {
    %cast = memref.cast %arg2 : memref<104x104xi32> to memref<104x104xi32, strided<[104, 1]>>
    call @kernel_matmul(%arg0, %arg1, %arg2) : (memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, memref<104x104xi32>) -> ()
    return
  }
  func.func @matmul_tiled_subviews(%arg0: memref<104x104xi8>, %arg1: memref<104x104xi8, strided<[1, 104]>>, %arg2: memref<104x104xi32>) attributes {llvm.emit_c_interface} {
    %cast = memref.cast %arg2 : memref<104x104xi32> to memref<104x104xi32, strided<[104, 1]>>
    call @tiled_matmul_w_subviews(%arg0, %arg1, %cast) : (memref<104x104xi8>, memref<104x104xi8, strided<[1, 104]>>, memref<104x104xi32, strided<[104, 1]>>) -> ()
    return
  }
}

