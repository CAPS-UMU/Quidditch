

#include "zigzag_dispatch.h"

// #include <assert.h>
// #include <encoding.h>
// #include <riscv_decls.h>
// #include <stdatomic.h>
// #include <stdbool.h>
// #include <stdint.h>
// #include <sync_decls.h>
// #include <team_decls.h>

#include <assert.h>
#include <cluster_interrupt_decls.h>
//#include <riscv.h>
#include <riscv_decls.h>
//#include <snitch_cluster_defs.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <team_decls.h>

//#include "../../../../../runtime/tests/lib-zigzag/memref.h"

// dispatch using spinlocks

// TODO: This should be cluster local.
static struct cluster_state_t {
  int bins[9];
  atomic_bool sleep[9];
  atomic_bool exit;
  //  void (*g)(void *a, void *b, void *c);
  void *a;
  void *b;
  void *c;
  kernel_ptr k[9];
  uint32_t a1;
  uint32_t b1;
  uint32_t c1;
  uint32_t c2;
  uint32_t a1_bk_sz;
  uint32_t b1_bk_sz;
  uint32_t c1_bk_sz;
  uint32_t c2_bk_sz;
  void *opI[9];
  void *opW[9];
  void *opO[9];
} cluster_state = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},                                      // bins
    {false, false, false, false, false, false, false, false, false},  // sleep
    false,                                                            // exit
    (kernel_ptr)0,                // kernel pointer g
    (void *)0,                    // a
    (void *)0,                    // b
    (void *)0,                    // c
    {0, 0, 0, 0, 0, 0, 0, 0, 0},  // kernel pointer k
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    {0, 0, 0, 0, 0, 0, 0, 0, 0},   // opI
    {0, 0, 0, 0, 0, 0, 0, 0, 0},   // opW
    {0, 0, 0, 0, 0, 0, 0, 0, 0}};  // opO

void set_kernel(uint32_t coreID,
                void (*g)(void *a, void *b, void *c, uint32_t a1, uint32_t b1,
                          uint32_t c1, uint32_t c2, uint32_t a1_bk_sz,
                          uint32_t b1_bk_sz, uint32_t c1_bk_sz,
                          uint32_t c2_bk_sz)) {
  // cluster_state.g = g;
  cluster_state.k[coreID] = g;
}

void save_outer_loop_counters(uint32_t a1_c, uint32_t b1_c, uint32_t c1_c,
                              uint32_t c2_c, uint32_t a1_bk_sz_c,
                              uint32_t b1_bk_sz_c, uint32_t c1_bk_sz_c,
                              uint32_t c2_bk_sz_c) {
  cluster_state.a1 = a1_c;
  cluster_state.b1 = b1_c;
  cluster_state.c1 = c1_c;
  cluster_state.c2 = c2_c;
  cluster_state.a1_bk_sz = a1_bk_sz_c;
  cluster_state.b1_bk_sz = b1_bk_sz_c;
  cluster_state.c1_bk_sz = c1_bk_sz_c;
  cluster_state.c2_bk_sz = c2_bk_sz_c;
}

void set_kernel_args(uint32_t coreID, void *a, void *b, void *c) {
  cluster_state.a = a;
  cluster_state.b = b;
  cluster_state.c = c;
  cluster_state.opI[coreID] = a;
  cluster_state.opW[coreID] = b;
  cluster_state.opO[coreID] = c;
}

void compute_core_loop() {
  while (!cluster_state.exit) {
    // sleep
    cluster_state.sleep[snrt_cluster_core_idx()] = true;
    while (cluster_state.sleep[snrt_cluster_core_idx()]) {
      ;
    }
    // If didn't get woken up to exit,
    if (!cluster_state.exit) {
      // do something
      // (*cluster_state.g)(cluster_state.a, cluster_state.b, cluster_state.c);
      (*cluster_state.k[snrt_cluster_core_idx()])(
          cluster_state.opI[snrt_cluster_core_idx()],
          cluster_state.opW[snrt_cluster_core_idx()],
          cluster_state.opO[snrt_cluster_core_idx()], cluster_state.a1,
          cluster_state.b1, cluster_state.c1, cluster_state.c2,
          cluster_state.a1_bk_sz, cluster_state.b1_bk_sz,
          cluster_state.c1_bk_sz, cluster_state.c2_bk_sz);
      cluster_state.bins[snrt_cluster_core_idx()]++;
    }
  }
}

void tell_compute_cores_to_exit() {
  cluster_state.exit = true;
  // cores can't cluster_state.exit until they wake up
  wake_up_compute_cores();
}

void wake_up_compute_core(uint32_t coreID) {
  cluster_state.sleep[coreID] = false;
}

void wake_up_compute_cores() {
  for (size_t i = 0; i < 8; i++) {
    cluster_state.sleep[i] = false;
  }
}

void wait_for_compute_core(uint32_t coreID) {
  while (!cluster_state.sleep[coreID]) {
  }
}

void wait_for_all_compute_cores() {
  //printf("waiting for compute cores...\n");
  while (!(cluster_state.sleep[0] && cluster_state.sleep[1] &&
           cluster_state.sleep[2] && cluster_state.sleep[3] &&
           cluster_state.sleep[4] && cluster_state.sleep[5] &&
           cluster_state.sleep[6] && cluster_state.sleep[7])) {
  }
  //printf("all the compute cores are done!\n");
}

void printBins() {
  printf("bins: %d %d %d %d %d %d %d %d %d\n", cluster_state.bins[0],
         cluster_state.bins[1], cluster_state.bins[2], cluster_state.bins[3],
         cluster_state.bins[4], cluster_state.bins[5], cluster_state.bins[6],
         cluster_state.bins[7], cluster_state.bins[8]);
}

uint32_t _mlir_ciface_myID() {
  return (uint32_t) snrt_cluster_core_idx();
}