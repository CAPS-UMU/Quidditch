
#pragma once

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "../../../../../runtime/tests/test_case_utils/memref.h"

// dispatch using spinlocks
// a1, b1, c1, c2, a1_bk_sz, b1_bk_sz, c1_bk_sz, c2_bk_sz
typedef void (*kernel_ptr)(void *arg0, void *arg1, void *arg2, uint32_t a1,
                           uint32_t b1, uint32_t c1, uint32_t c2,
                           uint32_t a1_bk_sz, uint32_t b1_bk_sz,
                           uint32_t c1_bk_sz, uint32_t c2_bk_sz);

extern void _mlir_ciface_mlirFunc(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                  TwoDMemrefI32_t *c);

// select the kernel the compute cores should execute
void set_kernel(uint32_t coreID, void (*g)(void *a, void *b, void *c, uint32_t a1,
                           uint32_t b1, uint32_t c1, uint32_t c2,
                           uint32_t a1_bk_sz, uint32_t b1_bk_sz,
                           uint32_t c1_bk_sz, uint32_t c2_bk_sz));
// todo: documentation
void save_outer_loop_counters(uint32_t a1_c, uint32_t b1_c, uint32_t c1_c,
                              uint32_t c2_c, uint32_t a1_bk_sz_c,
                              uint32_t b1_bk_sz_c, uint32_t c1_bk_sz_c,
                              uint32_t c2_bk_sz_c);

// provide valid addresses for the kernel's arguments
void set_kernel_args(uint32_t coreID, void *a, void *b, void *c);

// busy wait until DMA core says to exit or perform a computation
void compute_core_loop(void);

void tell_compute_cores_to_exit(void);
void wake_up_compute_cores(void);
void wake_up_compute_core(uint32_t coreID);

// wait for all compute cores to finish computing and return to "sleep"
void wait_for_all_compute_cores(void);

// wait for a particular compute core to finish computing and return to "sleep"
void wait_for_compute_core(uint32_t coreID);

// for debugging
void printBins(void);

uint32_t _mlir_ciface_myID() ;
