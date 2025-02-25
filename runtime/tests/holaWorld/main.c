// adapted from
// https://github.com/KULeuven-MICAS/snax-mlir/blob/f651860981efe0da84c0e5231bfcb03faf16890a/kernels/simple_matmul/main.c
// and
// https://github.com/EmilySillars/Quidditch-zigzag/blob/tiling/runtime/tests/tiledMatmul12/main.c

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

// Tiny computation to run on compute core #5
// - only arg0 is used
// - the rest of the arguments are left over from a 
// tiled matmul example this code was based on
void tiny(TwoDMemrefI8_t *arg0, TwoDMemrefI8_t *arg1, TwoDMemrefI32_t *arg2,
          uint32_t a1, uint32_t b1, uint32_t c1, uint32_t c2, uint32_t a1_bk_sz,
          uint32_t b1_bk_sz, uint32_t c1_bk_sz, uint32_t c2_bk_sz){
             void* weDontWantAMemrefStruct = (void*) arg0;
             int* num = (int*) weDontWantAMemrefStruct;
             *num = 2*snrt_cluster_core_idx();
}


int main() {
  if (!snrt_is_dm_core()){
    // if not the DMA core, enter the compute core loop.
    compute_core_loop();
    return 0;
  }

  int sum = 0; // stack allocated stuff is in L3

  printf("DMA core here with id %d \n", snrt_cluster_core_idx());

  set_kernel(5, (kernel_ptr)tiny);
  set_kernel_args(5, (void *)&sum, (void *)0, (void *)0); // we don't care about the 2nd and 3rd args in this case.
  wake_up_compute_core(5);  // when the compute core wakes up, it will execute the kernel tiny right away.
  wait_for_compute_core(5); // wait for CC to finish running tiny.

  int nerr =(sum == 10) ? 0 : 1;

  printf("%d\n", sum);

  tell_compute_cores_to_exit();

  return nerr;
}

