#include "fakeNN_util.h"

#include <Quidditch/dispatch/dispatch.h>

#include <iree/base/alignment.h>

#include <team_decls.h>
#include <util/run_model.h>

// ../toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/fakeNN/fakeNN

iree_status_t compiled_fake_n_n_create(iree_vm_instance_t *, iree_allocator_t,
                                      iree_vm_module_t **);

// copied from Quidditch/snitch_cluster/sw/snRuntime/src/riscv.h
inline uint32_t snrt_mcycle() {
  uint32_t register r;
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
  return r;
}

extern unsigned long myrtle_actual_cycles[5][2];


int run_fakeNN_experiment(
    iree_hal_executable_library_query_fn_t implementation) {
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  double(*inputData)[2*40] = aligned_alloc(64, (2*40) * sizeof(double));
  double(*outputData)[2*120] = aligned_alloc(64, (2*120) * sizeof(double));
    
  // initialize input data
  for (int i = 0; i < IREE_ARRAYSIZE(*inputData); i++) {
    (*inputData)[i] = (i + 1);
  }

  // initialize output data to all zeros
  for (int i = 0; i < IREE_ARRAYSIZE(*outputData); i++) {
    (*outputData)[i] = 0;
  }

  model_config_t config = {
      .libraries = (iree_hal_executable_library_query_fn_t[]){implementation},
      .num_libraries = 1,
      .module_constructor = compiled_fake_n_n_create,
      .main_function = iree_make_cstring_view("compiled_fake_n_n.main"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 1,
      .input_data = (const void *[]){inputData, inputData},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(*inputData)},
      .input_ranks = (const iree_host_size_t[]){3},
      .input_shapes = (const iree_hal_dim_t *[]){(iree_hal_dim_t[]){1, 2, 40}},

      .num_outputs = 1,
      .output_data = (void *[]){outputData},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(*outputData)},
  };

  unsigned long cycles = snrt_mcycle();
  IREE_CHECK_OK(run_model(&config));
  unsigned long cycles_after = snrt_mcycle();

  for (int i = 0; i < IREE_ARRAYSIZE(*outputData); i++) {
    double value = (*outputData)[i];
    printf("%f\n", value);
  }

  int i = 0;
  printf("dispatch 9: %lu - %lu = %lu\n", myrtle_actual_cycles[i][1],myrtle_actual_cycles[i][0],myrtle_actual_cycles[i][1]-myrtle_actual_cycles[i][0]); 
  i = 1;
  printf("dispatch 0: %lu - %lu = %lu\n", myrtle_actual_cycles[i][1],myrtle_actual_cycles[i][0],myrtle_actual_cycles[i][1]-myrtle_actual_cycles[i][0]); 
  i = 2;
  printf("dispatch 7: %lu - %lu = %lu\n", myrtle_actual_cycles[i][1],myrtle_actual_cycles[i][0],myrtle_actual_cycles[i][1]-myrtle_actual_cycles[i][0]); 
  i = 3;
  printf("dispatch 8: %lu - %lu = %lu\n", myrtle_actual_cycles[i][1],myrtle_actual_cycles[i][0],myrtle_actual_cycles[i][1]-myrtle_actual_cycles[i][0]); 
  i = 4;
  printf("dispatch 1: %lu - %lu = %lu\n", myrtle_actual_cycles[i][1],myrtle_actual_cycles[i][0],myrtle_actual_cycles[i][1]-myrtle_actual_cycles[i][0]); 
  

  printf("\ncycles %lu\n", cycles_after - cycles);
  free(inputData);
  free(outputData);
  return 0;
}
