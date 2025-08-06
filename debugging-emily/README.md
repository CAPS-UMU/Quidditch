# Eventually, delete this file and containing directory...
Debugging segfaults encountered when running many GrapeFruits...

1. Re-run segfaulting schemes for
   - dispatch 600x400
   - dispatch 600x600
2. Check fakeNN results (search for segfaults!!!)

## Which tiling schemes cause segfaults?

### 600x400

```
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-8/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-10/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-16/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-20/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-25/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-40/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-50/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-80/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-100/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-24-200/GrapeFruit
another batch...
1 1x600x400wm-n-k/0-40-8/GrapeFruit SEGFAULTS
2 1x600x400wm-n-k/0-40-10/GrapeFruit
3 1x600x400wm-n-k/0-40-16/GrapeFruit
4 1x600x400wm-n-k/0-40-20/GrapeFruit
5 1x600x400wm-n-k/0-40-25/GrapeFruit SEGFAULTS
6 1x600x400wm-n-k/0-40-40/GrapeFruit SEGFAULTS
7 1x600x400wm-n-k/0-40-50/GrapeFruit SEGFAULTS
8 1x600x400wm-n-k/0-40-80/GrapeFruit
9 1x600x400wm-n-k/0-40-100/GrapeFruit SEGFAULTS
10 1x600x400wm-n-k/0-120-8/GrapeFruit
another batch...
1 1x600x400wm-n-k/0-120-10/GrapeFruit
2 1x600x400wm-n-k/0-120-16/GrapeFruit
3 1x600x400wm-n-k/0-120-20/GrapeFruit
4 1x600x400wm-n-k/0-120-25/GrapeFruit
5 1x600x400wm-n-k/0-120-40/GrapeFruit SEGFAULTS
6 1x600x400wm-n-k/0-200-8/GrapeFruit
7 1x600x400wm-n-k/0-200-10/GrapeFruit
8 1x600x400wm-n-k/0-200-16/GrapeFruit
9 1x600x400wm-n-k/0-200-20/GrapeFruit
10 1x600x400wm-n-k/0-200-25/GrapeFruit SEGFAULTS
```

Let's create a new CSV of only runs that segfaulted:

```
1 1x600x400wm-n-k/0-40-8/GrapeFruit SEGFAULTS
5 1x600x400wm-n-k/0-40-25/GrapeFruit SEGFAULTS
6 1x600x400wm-n-k/0-40-40/GrapeFruit SEGFAULTS
7 1x600x400wm-n-k/0-40-50/GrapeFruit SEGFAULTS
9 1x600x400wm-n-k/0-40-100/GrapeFruit SEGFAULTS
5 1x600x400wm-n-k/0-120-40/GrapeFruit SEGFAULTS
10 1x600x400wm-n-k/0-200-25/GrapeFruit SEGFAULTS
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-600-8/GrapeFruit (who knows!)
```



### 600x600

```
1 1x600x600wm-n-k/0-40-20/GrapeFruit
2 1x600x600wm-n-k/0-40-24/GrapeFruit SEGFAULTS
3 1x600x600wm-n-k/0-40-25/GrapeFruit SEGFAULTS
4 1x600x600wm-n-k/0-40-30/GrapeFruit
5 1x600x600wm-n-k/0-40-40/GrapeFruit SEGFAULTS
6 1x600x600wm-n-k/0-40-50/GrapeFruit SEGFAULTS
7 1x600x600wm-n-k/0-40-60/GrapeFruit
8 1x600x600wm-n-k/0-40-75/GrapeFruit
9 1x600x600wm-n-k/0-40-100/GrapeFruit SEGFAULTS
10 1x600x600wm-n-k/0-40-120/GrapeFruit
another batch...
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-8/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-10/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-12/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-15/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-20/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-24/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-25/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-30/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-40/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-8/GrapeFruit
another batch... vvv unclear if any of these segfaulted or not vvvvvvvvvvvvvvvvvvvv
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-10/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-12/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-15/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-20/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-24/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-25/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-600-8/GrapeFruit
```

Let's create a new CSV of only runs that segfaulted:

```
2 1x600x600wm-n-k/0-40-24/GrapeFruit SEGFAULTS
3 1x600x600wm-n-k/0-40-25/GrapeFruit SEGFAULTS
5 1x600x600wm-n-k/0-40-40/GrapeFruit SEGFAULTS
6 1x600x600wm-n-k/0-40-50/GrapeFruit SEGFAULTS
9 1x600x600wm-n-k/0-40-100/GrapeFruit SEGFAULTS
```

Let's create a CSV of runs where we aren't sure if they segfaulted or not:

```
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-8/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-10/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-12/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-15/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-20/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-24/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-25/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-30/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-120-40/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-8/GrapeFruit
another batch... vvv unclear if any of these segfaulted or not vvvvvvvvvvvvvvvvvvvv
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-10/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-12/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-15/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-20/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-24/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-200-25/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x600wm-n-k/0-600-8/GrapeFruit
```



## If I remove the problem tiling schemes from the timing results, how does the cost model do?

0-40-100 segfaulted both times, so hard to say...

## Let's re-run and see if we get segfaults again!

```
. run_experiment.sh "1x600x400wm-n-k_case1_searchSpace_segfaults.csv" "1x600x400wm-n-k" genJsons no no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

comparing-tiling-schemes/1x600x400wm-n-k_case1_searchSpace_segfaults.csv

```
. run_experiment.sh "1x600x600wm-n-k_case1_searchSpace_segfaults.csv" "1x600x600wm-n-k" genJsons no no no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
```

Problem: I re-ran one that I thought would segfault, but it finished!

```
. run_experiment.sh "1x600x400wm-n-k_case1_searchSpace_segfaults_one.csv" "1x600x400wm-n-k" no no run no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

Let's re-run the entire 600x400 searchspace:



## looking at actual matmul lowering

#### how to define a model configt for a matmul with TWO inputs?????

bigmatvec:

```
  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_big_matvec_linked_quidditch_library_query,
          },
      .num_libraries = 1,
      .module_constructor = big_matvec_create,

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 2,
      .input_sizes = (const iree_host_size_t[]){1 * 320, 320 * 400},
      .input_ranks = (const iree_host_size_t[]){2, 2},
      .input_shapes = (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){1, 400},
                                                (iree_hal_dim_t[]){320, 400}},

      .num_outputs = 1,
      .output_sizes = (const iree_host_size_t[]){1, 320},
  };
```

grapefruit:

```

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
```

