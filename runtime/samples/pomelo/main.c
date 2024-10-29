#include <Quidditch/dispatch/dispatch.h>

#include <pamplemousse.h>
#include <pamplemousse_module.h>
#include <team_decls.h>
#include <util/run_model.h>

int main() {

  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();
  /*
  Let's start with something super simple - a [2x3] * [3x2] = [2x2]matmul:
  
  */

  // allocate space for input matrices a and b, and output matrix c.
  double(*a)[6] = aligned_alloc(64, 6 * sizeof(double));
  double(*b)[6] = aligned_alloc(64, 6 * sizeof(double));
  double(*c)[4] = aligned_alloc(64, 4 * sizeof(double));
  double(*golden)[4] = aligned_alloc(64, 4 * sizeof(double));
  // initialize input matrix a
  (*a)[0] = 1;
  (*a)[1] = 2;
  (*a)[2] = 3;
  (*a)[3] = 2;
  (*a)[4] = 2;
  (*a)[5] = 2;
  // initialize input matrix b
  (*b)[0] = 1;
  (*b)[1] = 1;
  (*b)[2] = 4;
  (*b)[3] = 2;
  (*b)[4] = 4;
  (*b)[5] = 3;
  // initialize output to all zeroes
  for (int i = 0; i < IREE_ARRAYSIZE(*c); i++) {
    (*c)[i] = 0;
  }
  // initialize golden
  (*golden)[0] = 21;
  (*golden)[1] = 14;
  (*golden)[2] = 18;
  (*golden)[3] = 12;

  
  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_matmulTiny_dispatch_0_library_query,
          },
      .num_libraries = 1,
      .module_constructor = test_pamplemousse_create,
      .main_function = iree_make_cstring_view("test_pamplemousse.matmulTiny"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 3,
      .input_data = (const void*[]){a, b, c},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(*a),
                                                IREE_ARRAYSIZE(*b),
                                                IREE_ARRAYSIZE(*c)},
      .input_ranks = (const iree_host_size_t[]){2, 2, 2},
      .input_shapes =
          (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){2,3},
                                    (iree_hal_dim_t[]){3,2},
                                    (iree_hal_dim_t[]){2,2}},

      .num_outputs = 1,
      .output_data = (void*[]){*c},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(*c)},
  };

  IREE_CHECK_OK(run_model(&config));

  if (!snrt_is_dm_core()) return 0;

  printf("hello there\n");
  for (int i = 0; i < IREE_ARRAYSIZE(*c); i++) {
    double value = (*c)[i];
    printf("%f\n", value);
    if (value != (*golden)[i]) return 1;
  }

  free(a);
  free(b);
  free(c);
  free(golden);
  printf("goodbye\n");
  return 0;
}
