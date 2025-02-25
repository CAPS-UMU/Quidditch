// adapted from
// https://github.com/KULeuven-MICAS/snax-mlir/blob/f651860981efe0da84c0e5231bfcb03faf16890a/kernels/simple_matmul/main.c

#include <Quidditch/zigzag_dispatch/zigzag_dispatch.h>

#include <assert.h>
#include <cluster_interrupt_decls.h>
#include <riscv_decls.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <team_decls.h>

#include "../test_case_utils/zigzag_utils.h"

// Kernel provided via external definition

extern void _mlir_ciface_matmul(TwoDMemrefI8_t *arg0, TwoDMemrefI8_t *arg1,
                                TwoDMemrefI32_t *arg2, uint32_t a1, uint32_t b1,
                                uint32_t c1, uint32_t c2, uint32_t a1_bk_sz,
                                uint32_t b1_bk_sz, uint32_t c1_bk_sz,
                                uint32_t c2_bk_sz);

int main() {
  if (!snrt_is_dm_core()) {
    compute_core_loop();
    return 0;
  }

  // printf("I am the DMA core with id %d", snrt_cluster_core_idx());

  // Create memref objects for data stored in L3
  TwoDMemrefI8_t memrefA;
  memrefA.data = (int8_t *)malloc(sizeof(int8_t) * MAT_WIDTH_SQUARED);
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  TwoDMemrefI8_t memrefB;
  memrefB.data = (int8_t *)malloc(sizeof(int8_t) * MAT_WIDTH_SQUARED);
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  TwoDMemrefI32_t memrefC;
  memrefC.data = (int32_t *)malloc(sizeof(int32_t) * MAT_WIDTH_SQUARED);
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;
  TwoDMemrefI32_t memrefGolden;
  memrefGolden.data = (int32_t *)malloc(sizeof(int32_t) * MAT_WIDTH_SQUARED);
  memrefGolden.aligned_data = memrefGolden.data;
  memrefGolden.offset = 0;

  // initialize the matrices
  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefA.aligned_data[i] = (int8_t)2;
  }
  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefB.aligned_data[i] = (int8_t)3;
  }

  // perform C code matmul to get the ground truth
  cCodeSquareMatmul(&memrefA, &memrefB, &memrefGolden);

  // perform matmul on compute core #5
  set_kernel(5, (kernel_ptr)_mlir_ciface_matmul);
  set_kernel_args(5, (void *)&memrefA, (void *)&memrefB, (void *)&memrefC);
  wake_up_compute_core(5);
  wait_for_compute_core(5);

  // check for correctness
  int nerr = 0;
  for (int i = 0; i < MAT_WIDTH_SQUARED; i++) {
    int32_t error = memrefC.aligned_data[i] - memrefGolden.aligned_data[i];
    if (error != 0) {
      nerr += 1;
      printf(" i is %d and %d /= %d\n", i, memrefC.aligned_data[i],
             memrefGolden.aligned_data[i]);
      break;
    }
  }

  if (nerr != 0) {
    printf("Output does not match the golden value!\n");
  } else {
    printf("Output Correct\n");
  }

  // free everything before exiting!
  free(memrefA.data);
  free(memrefB.data);
  free(memrefC.data);
  free(memrefGolden.data);

  // tell all compute cores (really just compute core 5) to exit
  tell_compute_cores_to_exit();
  return nerr;
}
