#include "grapeFruit_util.h"

#include <Quidditch/dispatch/dispatch.h>

#include <iree/base/alignment.h>

#include <team_decls.h>
#include <util/run_model.h>

// #include <snrt.h>
///home/hoppip/Quidditch/runtime/samples/grapeFruit/grapeFruit_util.c"
///home/hoppip/Quidditch/runtime/snitch_cluster/rtl/src/snrt.h

iree_status_t compiled_ns_net2_create(iree_vm_instance_t *, iree_allocator_t,
                                      iree_vm_module_t **);

unsigned long read_cycle_direct()
{
  unsigned long cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

inline uint32_t snrt_mcycle() {
    uint32_t register r;
    asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
    return r;
}

// main () {
// ..... 
//   unsigned long cycles = read_cycle_direct();
//   OMTensorList *tensorListOut = run_main_graph(tensorListIn);
//   unsigned long cycles_after = read_cycle_direct();
//   std::cout << "Cycles: " << cycles_after - cycles << std::endl;

// ....
// }
int run_grapeFruit_experiment(
    iree_hal_executable_library_query_fn_t implementation) {
  if (!snrt_is_dm_core()) return 0;//quidditch_dispatch_enter_worker_loop();

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

  //snrt_mcycle();
  unsigned long cycles = snrt_mcycle();//read_cycle_direct();
 // IREE_CHECK_OK(run_model(&config));
 

  // for (int i = 0; i < IREE_ARRAYSIZE(*data); i++) {
  //   double value = (*data)[i];
  //   printf("%f\n", value);
  // }
  printf("hoodle\n");
  unsigned long cycles_after = snrt_mcycle();//read_cycle_direct();
  //snrt_mcycle();
  printf("cycles %lu\n", cycles_after - cycles);
  free(data);
  return 0;
}
