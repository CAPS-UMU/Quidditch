#include "nsnet2_util.h"

#include <Quidditch/dispatch/dispatch.h>
#include <Quidditch/time_dispatch/time_dispatch.h>
#include <iree/base/alignment.h>

#include <team_decls.h>
#include <util/run_model.h>

iree_status_t compiled_ns_net2_create(iree_vm_instance_t *, iree_allocator_t,
                                      iree_vm_module_t **);

// copied from Quidditch/snitch_cluster/sw/snRuntime/src/riscv.h
inline uint32_t snrt_mcycle() {
  uint32_t register r;
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
  return r;
}

extern unsigned long myrtle_actual_cycles[5][2];

int run_nsnet2_experiment(
    iree_hal_executable_library_query_fn_t implementation) {
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  double(*data)[161] = aligned_alloc(64, 161 * sizeof(double));

  for (int i = 0; i < IREE_ARRAYSIZE(*data); i++) {
    (*data)[i] = (i + 1);
  }

  model_config_t config = {
      .libraries = (iree_hal_executable_library_query_fn_t[]){implementation},
      .num_libraries = 1,
      .module_constructor = compiled_ns_net2_create,
      .main_function = iree_make_cstring_view("compiled_ns_net2.main"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 1,
      .input_data = (const void *[]){data, data},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(*data)},
      .input_ranks = (const iree_host_size_t[]){3},
      .input_shapes = (const iree_hal_dim_t *[]){(iree_hal_dim_t[]){1, 1, 161}},

      .num_outputs = 1,
      .output_data = (void *[]){data},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(*data)},
  };

  unsigned long cycles = snrt_mcycle();
  IREE_CHECK_OK(run_model(&config));
  unsigned long cycles_after = snrt_mcycle();

  for (int i = 0; i < IREE_ARRAYSIZE(*data); i++) {
    double value = (*data)[i];
    printf("%f\n", value);
  }
   for(int i = 0; i < 5; i ++){
    printf("%d: %lu - %lu = %lu\n", i, myrtle_actual_cycles[i][1],myrtle_actual_cycles[i][0],myrtle_actual_cycles[i][1]-myrtle_actual_cycles[i][0]);     
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
  free(data);
  return 0;
}
