#include <Quidditch/dispatch/dispatch.h>

#include <pamplemousse.h>
#include <pamplemousse_module.h>
#include <team_decls.h>
#include <util/run_model.h>

int main() {
  iree_alignas(64) double data[4];
  // statically allocate space for input/output and solution matrices.
  iree_alignas(64) double a[6];
  iree_alignas(64) double b[6];
  iree_alignas(64) double c[4];
  iree_alignas(64) double golden[4];
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  // initialize input matrix a
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 2;
  a[4] = 2;
  a[5] = 2;
  // initialize input matrix b
  b[0] = 1;
  b[1] = 1;
  b[2] = 4;
  b[3] = 2;
  b[4] = 4;
  b[5] = 3;
  // initialize output to all zeroes
  for (int i = 0; i < IREE_ARRAYSIZE(c); i++) {
    c[i] = 0;
  }
  // initialize golden
  golden[0] = 21;
  golden[1] = 14;
  golden[2] = 18;
  golden[3] = 12;

  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    data[i] = (i + 1);
  }

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_pamplemousse_linked_quidditch_library_query,
          },
      .num_libraries = 1,
      .module_constructor = pamplemousse_create,
      // .main_function = iree_make_cstring_view("pamplemousse.add"),

      // .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      // .num_inputs = 2,
      // .input_data = (const void*[]){data, data},
      // .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data),
      //                                           IREE_ARRAYSIZE(data)},
      // .input_ranks = (const iree_host_size_t[]){1, 1},
      // .input_shapes =
      //     (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){IREE_ARRAYSIZE(data)},
      //                               (iree_hal_dim_t[]){IREE_ARRAYSIZE(data)}},

      // .num_outputs = 1,
      // .output_data = (void*[]){data},
      // .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data)},
      // matmulTiny
      .main_function = iree_make_cstring_view("pamplemousse.matmulTiny"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 3,
      .input_data = (const void*[]){a, b, c},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(a),
                                                IREE_ARRAYSIZE(b),
                                                IREE_ARRAYSIZE(c)},
      .input_ranks = (const iree_host_size_t[]){2, 2, 2},
      .input_shapes =
          (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){2, 3},
                                    (iree_hal_dim_t[]){3, 2},
                                    (iree_hal_dim_t[]){2, 2}},

      .num_outputs = 1,
      .output_data = (void*[]){c},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(c)},
  };

  IREE_CHECK_OK(run_model(&config));

  if (!snrt_is_dm_core()) return 0;

  for (int i = 0; i < IREE_ARRAYSIZE(c); i++) {
    double value = c[i];
    printf("%f\n", value);
    if (value != golden[i]) return 1;
  }
  return 0;
}

/*
[fesvr] Wrote 36 bytes of bootrom to 0x1000
[fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
[fesvr] Wrote 56 bytes of bootdata to 0x1024
[Tracer] Logging Hart          8 to logs/trace_hart_00000008.dasm
[Tracer] Logging Hart          0 to logs/trace_hart_00000000.dasm
[Tracer] Logging Hart          1 to logs/trace_hart_00000001.dasm
[Tracer] Logging Hart          2 to logs/trace_hart_00000002.dasm
[Tracer] Logging Hart          3 to logs/trace_hart_00000003.dasm
[Tracer] Logging Hart          4 to logs/trace_hart_00000004.dasm
[Tracer] Logging Hart          5 to logs/trace_hart_00000005.dasm
[Tracer] Logging Hart          6 to logs/trace_hart_00000006.dasm
[Tracer] Logging Hart          7 to logs/trace_hart_00000007.dasm
RESOURCE_EXHAUSTED; while invoking native function pamplemousse.matmulTiny; 
[ 1]   native hal.fence.create:0 -
[ 0]   native pamplemousse.matmulTiny:0 -

*/
