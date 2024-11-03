#include <Quidditch/dispatch/dispatch.h>

#include <pumpkin.h>
#include <pumpkin_module.h>
#include <team_decls.h>
#include <util/run_model.h>

int main() {

  // statically allocate space for input/output and solution matrices.
  iree_alignas(64) double a[64];
  iree_alignas(64) double b[64];
  iree_alignas(64) double c[64];
  iree_alignas(64) double golden[64];
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  // initialize matrices
  for (int i = 0; i < IREE_ARRAYSIZE(c); i++) {
    a[i] = 2;
    b[i] = 3;
    c[i] = 0;
    golden[i] = 147;
  }

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_matmulTiny_dispatch_0_library_query,
          },
      .num_libraries = 1,
      .module_constructor = pumpkin_create,

      .main_function = iree_make_cstring_view("pumpkin.matmulTiny"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 3,
      .input_data = (const void*[]){a, b, c},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(a),
                                                IREE_ARRAYSIZE(b),
                                                IREE_ARRAYSIZE(c)},
      .input_ranks = (const iree_host_size_t[]){2, 2, 2},
      .input_shapes =
          (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){8, 8},
                                    (iree_hal_dim_t[]){8, 8},
                                    (iree_hal_dim_t[]){8, 8}},

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
