#include "zigzag_utils.h"

void set_accelerator_computation(uint32_t accID, kernel_ptr k) {
  set_kernel(accID, k);
}

void host_acc_perform_kernel_together(kernel_ptr k, void *arg0, void *arg1,
                                      void *arg2, void *arg3) {
  if (k == (kernel_ptr)_mlir_ciface_tiled_matmul) {
    _mlir_ciface_tiled_matmul(arg0, arg1, arg2, arg3);
  } else if (k == (kernel_ptr)_mlir_ciface_dummy) {
    _mlir_ciface_dummy(arg0, arg1, arg2, arg3);
  }
}

void host_acc_perform_kernel_together_2_slices(kernel_ptr k, void *arg0,
                                               void *arg1, void *arg2,
                                               void *slice1, void *slice2) {
  // TODO: fix so this uses pinapple
  _mlir_ciface_pineapple(arg0, arg1, arg2, slice1, slice2);
  //_mlir_ciface_tiled_matmul_2_slices(arg0, arg1, arg2, slice1, slice2);
}

void _mlir_ciface_save_outer_loop_counters(uint32_t a1_c, uint32_t b1_c,
                                           uint32_t c1_c, uint32_t c2_c,
                                           uint32_t a1_bk_sz_c,
                                           uint32_t b1_bk_sz_c,
                                           uint32_t c1_bk_sz_c,
                                           uint32_t c2_bk_sz_c) {
  save_outer_loop_counters(a1_c, b1_c, c1_c, c2_c, a1_bk_sz_c, b1_bk_sz_c,
                           c1_bk_sz_c, c2_bk_sz_c);
}

void _mlir_ciface_dispatch_to_accelerator(uint32_t accID, TwoDMemrefI8_t *arg0,
                                          TwoDMemrefI8_t *arg1,
                                          TwoDMemrefI32_t *arg2) {
  set_kernel_args(accID, arg0, arg1, arg2);
  // perform tiled matmul on compute core # accID
  wake_up_compute_core(accID);
}

void _mlir_ciface_wait_for_accelerator(uint32_t accID) {
  wait_for_compute_core(accID);
}

void _mlir_ciface_wait_for_all_accelerators() { wait_for_all_compute_cores(); }

// A C function accessible to MLIR
void _mlir_ciface_hola(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                       TwoDMemrefI32_t *c) {
  printf("hola world!\n");
}

void _mlir_ciface_memrefCopy8bit(TwoDMemrefI8_t *src, TwoDMemrefI8_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + (dst->stride[0] * row) +
                        (col * dst->stride[1])] =
          src->aligned_data[src->offset + (src->stride[0] * row) +
                            (col * src->stride[1])];
    }
  }
}

void _mlir_ciface_memrefCopy8bit_I_104x104(TwoDMemrefI8_t *src,
                                           TwoDMemrefI8_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + (dst->stride[0] * row) +
                        (col * dst->stride[1])] =
          src->aligned_data[src->offset + (src->stride[0] * row) +
                            (col * src->stride[1])];
    }
  }
}

void _mlir_ciface_memrefCopy8bit_W_104x104(TwoDMemrefI8_t *src,
                                           TwoDMemrefI8_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      // printf("I see the value %d",dst->aligned_data[dst->offset +
      // (dst->stride[0] * row) +
      //                   (col * dst->stride[1])]);
      dst->aligned_data[dst->offset + (dst->stride[0] * row) +
                        (col * dst->stride[1])] =
          src->aligned_data[src->offset + (src->stride[0] * row) +
                            (col * src->stride[1])];
    }
  }
}

void _mlir_ciface_memrefCopy8bit_W_26x104(TwoDMemrefI8_t *src,
                                          TwoDMemrefI8_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      // printf("I see the value %d",dst->aligned_data[dst->offset +
      // (dst->stride[0] * row) +
      //                   (col * dst->stride[1])]);
      // printf("I'm about to copy over %d",src->aligned_data[src->offset +
      // (src->stride[0] * row) +
      //                   (col * src->stride[1])]);
      // do the copy
      dst->aligned_data[dst->offset + (dst->stride[0] * row) +
                        (col * dst->stride[1])] =
          src->aligned_data[src->offset + (src->stride[0] * row) +
                            (col * src->stride[1])];
    }
  }
}

void setValMemref8Bit(TwoDMemrefI8_t *src, int8_t val) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      size_t index =
          (src->offset) + (src->stride[0] * row) + (col * src->stride[1]);
      src->aligned_data[index] = val;
      // if(index >= src->shape[0]*src->shape[1]){
      //   printf("index is %d which is >= %d. row is %d col is %d. stride[0] is
      //   %d stride[1] is %d\n",index, src->shape[0]*src->shape[1], row, col,
      //   src->stride[0],src->stride[1]);

      // }
    }
  }
}

// void _mlir_ciface_print_memref_8_bit(TwoDMemrefI8_t *src) {
//   printf("printing memref with shape %d x %d, offset %d: stride:
//   [%d,%d]\n[",src->shape[0],src->shape[1],
//   src->offset,src->stride[0],src->stride[1]); for (size_t row = 0; row <
//   src->shape[0]; row++) {
//     for (size_t col = 0; col < src->shape[1]; col++) {
//       // printf("I see the value %d",dst->aligned_data[dst->offset +
//       // (dst->stride[0] * row) +
//       //                   (col * dst->stride[1])]);
//       printf(" %d ",src->aligned_data[src->offset +
//       (src->stride[0] * row) +
//                         (col * src->stride[1])]);
//     }
//     printf("\n");
//   }
//   //  for (size_t row = 0; row < 104; row++) {
//   //   for (size_t col = 0; col < 104; col++) {
//   //     // printf("I see the value %d",dst->aligned_data[dst->offset +
//   //     // (dst->stride[0] * row) +
//   //     //                   (col * dst->stride[1])]);
//   //     printf(" %d ",(src->aligned_data)[row+(col*104)]);
//   //   }
//   //   printf("\n");
//   // }
//   // for(size_t j = 0; j < (src->shape[0]*src->shape[1]); j++){
//   //   printf(" %d ",src->aligned_data[j]);
//   // }
//   printf("]\n");
// }

// void _mlir_ciface_print_memref_8_bit(TwoDMemrefI8_t *src) {
//   printf("printing memref with shape %d x %d, offset %d: stride:
//   [%d,%d]\n[",src->shape[0],src->shape[1],
//   src->offset,src->stride[0],src->stride[1]);
// if(src->stride[0] == 1){
//   for(size_t j = 0; j < (src->shape[0]*src->shape[1]); j++){
//     if((j % src->stride[0]) == 0){
//       printf("\n %d ",src->aligned_data[j]);
//     }
//     else{
//        printf(" %d ",src->aligned_data[j]);
//     }
//   }
// }
// else{
//   for(size_t j = 0; j < (src->shape[0]*src->shape[1]); j++){
//     if((j % src->stride[0]) == 0){
//       printf("\n %d ",src->aligned_data[j]);
//     }
//     else{
//        printf(" %d ",src->aligned_data[j]);
//     }
//   }

// }
//   printf("]\n");
// }

void _mlir_ciface_print_memref_8_bit_python(char *name, TwoDMemrefI8_t *src) {
  printf("%s = [ ", name);
  size_t flat_size = src->shape[0] * src->shape[1];
  for (size_t i = 0; i < flat_size-1; i++) {
    printf(" %d, ", src->aligned_data[i]);
  }
  printf(" %d ", src->aligned_data[flat_size-1]);
  printf("]\n");
}

void _mlir_ciface_print_memref_32_bit_python(char *name, TwoDMemrefI32_t *src) {
  printf("%s = [ ", name);
  size_t flat_size = src->shape[0] * src->shape[1];
  for (size_t i = 0; i < flat_size-1; i++) {
    printf(" %d, ", src->aligned_data[i]);
  }
  printf(" %d ", src->aligned_data[flat_size-1]);
  printf("]\n");
}

void _mlir_ciface_print_memref_8_bit_FLAT(TwoDMemrefI8_t *src) {
  printf(
      "FLAT printing memref with shape %d x %d, offset %d: stride: [%d,%d]\n[",
      src->shape[0], src->shape[1], src->offset, src->stride[0],
      src->stride[1]);
  for (size_t i = 0; i < src->shape[0] * src->shape[1]; i++) {
    if (i % src->shape[0] == 0) {
      printf("\n %d ", src->aligned_data[i]);

    } else {
      printf(" %d ", src->aligned_data[i]);
    }
  }
  printf("]\n");
}

void _mlir_ciface_print_memref_8_bit_NOT_FLAT(TwoDMemrefI8_t *src) {
  printf(
      "NOT FLAT printing memref with shape %d x %d, offset %d: stride: "
      "[%d,%d]\n[",
      src->shape[0], src->shape[1], src->offset, src->stride[0],
      src->stride[1]);
  for (size_t i = 0; i < src->shape[0]; i++) {
    for (size_t j = 0; j < src->shape[1]; j++) {
      size_t index = i * (src->stride[0]) + j * (src->stride[1]);
      printf(" %d ", src->aligned_data[index]);
    }
    printf("\n");
  }
  printf("]\n");
}

void _mlir_ciface_print_memref_8_bit(TwoDMemrefI8_t *src) {
  printf("printing memref with shape %d x %d, offset %d: stride: [%d,%d]\n[",
         src->shape[0], src->shape[1], src->offset, src->stride[0],
         src->stride[1]);
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      printf(" %d ", src->aligned_data[src->offset + (src->stride[0] * row) +
                                       (col * src->stride[1])]);
    }
    printf("\n");
  }
  printf("]\n");
}

void _mlir_ciface_print_memref_32_bit(TwoDMemrefI32_t *src) {
  printf("printing memref with shape %d x %d, offset %d: stride: [%d,%d]\n[",
         src->shape[0], src->shape[1], src->offset, src->stride[0],
         src->stride[1]);
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      printf(" %d ", src->aligned_data[src->offset + (src->stride[0] * row) +
                                       (col * src->stride[1])]);
    }
    printf("\n");
  }
  printf("]\n");
}

void _mlir_ciface_print_index(int32_t i) {
  if (i == 624) {
    printf("\n");
  } else {
    printf("{ %d }", i);
  }
}

void _mlir_ciface_print_weight_elt(int32_t c, int32_t b, int32_t i) {
  if (_mlir_ciface_myID() == 0) {
    if (i != 3) {
      printf("w[%d][%d]=%d\n", c, b, i);
    }
  }
}

void _mlir_ciface_print_weight_elt2(int32_t c, int32_t b, int32_t i, int32_t a1,
                                    int32_t b0, int32_t b1, int32_t c0,
                                    int32_t c1, int32_t c2) {
  // if(_mlir_ciface_myID() == 0){
  if (i != 3) {
    printf("w[%d][%d]=%d\t, a1:%d b0:%d b1:%d c0:%d c1:%d c2:%d \n", c, b, i,
           a1, b0, b1, c0, c1, c2);
    // printf("a1:%d b0:%d b1:%d c0:%d c1:%d c2:%d \n",a1, b0, b1, c0, c1, c2);
  }
  // }
}

// void _mlir_ciface_print_weight_elt(int32_t c, int32_t b, int32_t i ) {

//   if(i != 3){
//     printf("w[%d][%d]=%d\n",c,b,i);
//   }

// }

void _mlir_ciface_print_i32(int32_t i) {
  if (i == 624) {
    printf("\n");
  } else {
    printf("={ %d }", i);
  }
}

void _mlir_ciface_memrefCopy32bit(TwoDMemrefI32_t *src, TwoDMemrefI32_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + dst->stride[0] * row +
                        col * dst->stride[1]] =
          src->aligned_data[src->offset + src->stride[0] * row +
                            col * src->stride[1]];
    }
  }
}

void _mlir_ciface_memrefCopy32bit_O_104x13(TwoDMemrefI32_t *src,
                                           TwoDMemrefI32_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + dst->stride[0] * row +
                        col * dst->stride[1]] =
          src->aligned_data[src->offset + src->stride[0] * row +
                            col * src->stride[1]];
    }
  }
}

void _mlir_ciface_memrefCopy32bit_O_8x8_down(TwoDMemrefI32_t *src,
                                             TwoDMemrefI32_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + dst->stride[0] * row +
                        col * dst->stride[1]] =
          src->aligned_data[src->offset + src->stride[0] * row +
                            col * src->stride[1]];
    }
  }
}

void _mlir_ciface_memrefCopy32bit_O_8x8_up(TwoDMemrefI32_t *src,
                                           TwoDMemrefI32_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + dst->stride[0] * row +
                        col * dst->stride[1]] =
          src->aligned_data[src->offset + src->stride[0] * row +
                            col * src->stride[1]];
    }
  }
}

void _mlir_ciface_memrefCopy32bit_O_104x1(TwoDMemrefI32_t *src,
                                          TwoDMemrefI32_t *dst) {
  for (size_t row = 0; row < src->shape[0]; row++) {
    for (size_t col = 0; col < src->shape[1]; col++) {
      dst->aligned_data[dst->offset + dst->stride[0] * row +
                        col * dst->stride[1]] =
          src->aligned_data[src->offset + src->stride[0] * row +
                            col * src->stride[1]];
    }
  }
}

/*
NEXT STEPS:
1) write intuitive matmul and compare with MAC matmul.
2) IF they give the SAME results, figure out why tiled matmul is WRONG.
3) IF they give different results, see if transposed weight matrix is the
issue.??
*/

// c-code implementation of matmul (to check correctness of MLIR)
void cCodeSquareMatmul(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y,
                       TwoDMemrefI32_t *z) {
  int z_index, x_index, y_index = 0;
  for (int d0 = 0; d0 < MAT_WIDTH; d0++) {
    for (int d1 = 0; d1 < MAT_WIDTH; d1++) {
      for (int d2 = 0; d2 < MAT_WIDTH; d2++) {
        // arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
        z_index = (d0 * MAT_WIDTH) + d1;
        x_index = (d0 * MAT_WIDTH) + d2;
        y_index = (d2 * MAT_WIDTH) + d1;
        z->aligned_data[z_index] +=
            x->aligned_data[x_index] * y->aligned_data[y_index];
      }
    }
  }
}

void cCodeSquareMatmul_intuitive(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                 TwoDMemrefI32_t *c) {
  int a_index, b_index, c_index = 0;
  for (int d0 = 0; d0 < MAT_WIDTH; d0++) {
    for (int d1 = 0; d1 < MAT_WIDTH; d1++) {
      // c[d0][d1] += a[d0][d1] * b[d1][d0]
      c_index = (d0 * MAT_WIDTH) + d1;
      a_index = c_index;
      b_index = (d1 * MAT_WIDTH) + d0;
      c->aligned_data[c_index] +=
          a->aligned_data[a_index] * b->aligned_data[b_index];
    }
  }
}

// printing functions
void print2DMemRefI8_t(TwoDMemrefI8_t *x, int32_t width) {
  printf("[\n");
  // we ASSUME a square 2D array
  int32_t col = 0;
  for (int i = 0; i < width * width; i++) {
    if (col == width) {
      col = 0;
      printf("\n %d ", x->aligned_data[i]);

    } else {
      printf(" %d ", x->aligned_data[i]);
    }
    col++;
  }
  printf("]\n");
}

void print2DMemRefI32_t(TwoDMemrefI32_t *x, int32_t width) {
  printf("[\n");
  // we ASSUME a square 2D array
  int32_t col = 0;
  for (int i = 0; i < width * width; i++) {
    if (col == width) {
      col = 0;
      printf("\n %d ", x->aligned_data[i]);

    } else {
      printf(" %d ", x->aligned_data[i]);
    }
    col++;
  }
  printf("]\n");
}

void print2DMemRefI32_t_notASquare(TwoDMemrefI32_t *x, int32_t stride_x,
                                   int32_t stride_y) {
  printf("[\n");
  int32_t col = 0;
  for (int i = 0; i < stride_x * stride_y; i++) {
    if (col == stride_x) {
      col = 0;
      printf("\n %d ", x->aligned_data[i]);

    } else {
      printf(" %d ", x->aligned_data[i]);
    }
    col++;
  }
  printf("]\n");
}