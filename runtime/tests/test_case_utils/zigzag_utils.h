// all the defintions needed for ZigZag integration
// we assume square matrices (for now)

#include <Quidditch/zigzag_dispatch/zigzag_dispatch.h>

#include "memref.h"  // from KULeuven's snax-mlir repo: https://github.com/KULeuven-MICAS/snax-mlir/blob/4666236ffd848eff1a4634c824ad257ba68d9a64/runtime/include/memref.h

// for now, we assume a square matrix
#ifndef MAT_WIDTH
#define MAT_WIDTH 104
#endif
#define MAT_WIDTH_SQUARED (MAT_WIDTH * MAT_WIDTH)

/*
All ZigZag sees is a memory hierarchy attached to a MAC array.
It recommends an optimal tiling scheme, but to implement it,
we need to carry out the correct movement of data tiles such that
everything is where it's supposed to be when each PE within the mac array runs.
To implement the correct data movement for ZigZag's schedule,
we employ a "host-accelerator" abstraction.
*/

// "Host" function calls

// Tell the accelerator(s) which computation it should perform.
//  Note: all computations (derived from kernels) have three arguments,
//  and are passed as function pointers
//  for now, all accelerators must perform the same computation
void set_accelerator_computation(uint32_t accID,
                                  kernel_ptr k);

// Launch kernel in host, which will dispatch computation to accelerator
void host_acc_perform_kernel_together(kernel_ptr k, void *arg0, void *arg1,
                                      void *arg2, void *arg3);
void host_acc_perform_kernel_together_2_slices(kernel_ptr k, void *arg0,
                                               void *arg1, void *arg2,
                                               void *slice1, void *slice2);

// dispatch workload to accelerator with id accID
void _mlir_ciface_dispatch_to_accelerator(uint32_t accID, TwoDMemrefI8_t *arg0,
                                          TwoDMemrefI8_t *arg1,
                                          TwoDMemrefI32_t *arg2);

void _mlir_ciface_mango(TwoDMemrefI8_t *arg0,
                        TwoDMemrefI8_t *arg1, TwoDMemrefI32_t *arg2);
void _mlir_ciface_mangoDummy(uint32_t accID, TwoDMemrefI8_t *arg0,
                        TwoDMemrefI8_t *arg1, TwoDMemrefI32_t *arg2);
// documentation TODO
void  _mlir_ciface_wait_for_accelerator(uint32_t accID);
void  _mlir_ciface_wait_for_all_accelerators();


// DL Kernels Defined in MLIR
extern void _mlir_ciface_dummy(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                               TwoDMemrefI32_t *c, TwoDMemrefI32_t *d);
extern void _mlir_ciface_tiled_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                      TwoDMemrefI32_t *c,
                                      TwoDMemrefI32_t *l1OTile);
extern void _mlir_ciface_pineapple(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                   TwoDMemrefI32_t *c, TwoDMemrefI32_t *OSlice,
                                   TwoDMemrefI8_t *WSlice);

extern void _mlir_ciface_tiledMatmul15(uint32_t coreID, TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                       TwoDMemrefI32_t *c,
                                       TwoDMemrefI8_t *sliceI,
                                       TwoDMemrefI8_t *sliceW,
                                       TwoDMemrefI32_t *sliceO);

// Helper functions for DL Kernels Defined in MLIR
extern void _mlir_ciface_matmul_accelerator_work(TwoDMemrefI8_t *arg0,
                                                 TwoDMemrefI8_t *arg1,
                                                 TwoDMemrefI32_t *arg2);

// Miscellaneous functions used by main.c

// compute matrix multiplication (for checking mlir matmul correctness)
void cCodeSquareMatmul(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y,
                       TwoDMemrefI32_t *z);
void cCodeSquareMatmul_intuitive(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                       TwoDMemrefI32_t *c);

// printing functions to help with debugging
void print2DMemRefI8_t(TwoDMemrefI8_t *x, int32_t width);
void print2DMemRefI32_t(TwoDMemrefI32_t *x, int32_t width);
void print2DMemRefI32_t_notASquare(TwoDMemrefI32_t *x, int32_t stride_x,
                                   int32_t stride_y);

// An external function implemented in MLIR
extern void _mlir_ciface_dummy(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                               TwoDMemrefI32_t *c, TwoDMemrefI32_t *d);
// A C function accessible to MLIR
void _mlir_ciface_hola(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                       TwoDMemrefI32_t *c);
void _mlir_ciface_memrefCopy8bit(TwoDMemrefI8_t *src, TwoDMemrefI8_t *dst);
void _mlir_ciface_memrefCopy8bit_I_104x104(TwoDMemrefI8_t *src, TwoDMemrefI8_t *dst);
void _mlir_ciface_memrefCopy8bit_W_104x104(TwoDMemrefI8_t *src, TwoDMemrefI8_t *dst);
void _mlir_ciface_memrefCopy8bit_W_26x104(TwoDMemrefI8_t *src, TwoDMemrefI8_t *dst);
void _mlir_ciface_memrefCopy32bit(TwoDMemrefI32_t *src, TwoDMemrefI32_t *dst);
void _mlir_ciface_memrefCopy32bit2(TwoDMemrefI32_t *src, TwoDMemrefI32_t *dst);
void _mlir_ciface_memrefCopy32bit_O_104x13(TwoDMemrefI32_t *src, TwoDMemrefI32_t *dst);
void _mlir_ciface_memrefCopy32bit_O_104x1(TwoDMemrefI32_t *src, TwoDMemrefI32_t *dst);
void _mlir_ciface_memrefCopy32bit_O_8x8(TwoDMemrefI32_t *src, TwoDMemrefI32_t *dst);
void _mlir_ciface_print_memref_8_bit(TwoDMemrefI8_t *src);
void _mlir_ciface_print_memref_32_bit(TwoDMemrefI32_t *src);
void setValMemref8Bit(TwoDMemrefI8_t *src, int8_t val);
void _mlir_ciface_print_memref_8_bit_FLAT(TwoDMemrefI8_t *src);
void _mlir_ciface_print_memref_8_bit_NOT_FLAT(TwoDMemrefI8_t *src);
void _mlir_ciface_print_memref_8_bit_python(char *name, TwoDMemrefI8_t *src);
void _mlir_ciface_print_memref_32_bit_python(char *name, TwoDMemrefI32_t *src);