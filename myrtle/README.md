# Myrtle Cost Model

## How to Test Myrtle with Different Tile Sizes

1. Navigate to the comparing-tile-sizes directory
   ```
   cd comparing-tile-sizes
   ```

2. Add tiling schemes (as json files) to the folder `tile-sizes-to-test`
### One Tiling Scheme

3. cmake and ninja build only one of the tiling schemes inside `tile-sizes-to-test` :
```
sh manyGrapefruits.sh actuallyOnlyOne 0-40-100.json noRun
```

4.  Run only one of the tiling schemes using verilator:

```
sh manyGrapefruits.sh actuallyOnlyOne 0-40-100.json
```
### All Tiling Schemes

3. Run cmake and ninja build for ***all tiling schemes inside*** `tile-sizes-to-test` with
   ```
   sh manyGrapefruits.sh cmake
   ```

4. Run each tiled nsnet one by one with
   ```
   sh manyGrapefruits.sh
   ```

5. Results will be in subfolders named after the json file specifying the tiling scheme. For example, a directory structure like
   ```
   comparing-tile-sizes/
   |---- 0-48-100/
         |---- logs/
         |---- buildOutput.txt
         |---- cmakeOutput.txt
         |---- GrapeFruit      
         |---- run_output.txt
         |---- tilingCosts.json
   ```

## Updated, Updated, Automated Flow

1. Export tiling schemes for a particular dispatch as a CSV with `filename here.py`

2. **Generate json** inputs, **compile** `nsnet2` with each tiling scheme, **run** `nsnet2` with each tiling scheme, and **export** cycle counts with

   ```
   sh run_experiment.sh <searchSpace.csv> <outputFolderName> <generateJSONs?> <compile?> <run?> <export?> <dispatchNo> "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
   ```

   where
   

### Shortcut Commands

Only Export results:

```
sh run_experiment.sh "dispatch_1_case3_searchSpace-export-early.csv" dispatch_1_case_3 no no no export 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```



## Updated, Automated Flow

### A Set of Tiling Schemes (from a CSV file)

1. Export generated tiling schemes from google colab notebook as a csv [here](https://colab.research.google.com/drive/1Vk24yIjoPN01qXXHLcGuL86TQHyKxYgx?usp=sharing)

2. **Generate json** inputs, **compile** `nsnet2` with each tiling scheme, **run** `nsnet2` with each tiling scheme, and **export** cycle counts with

   ```
   sh run_experiment.sh "case1_searchSpace.csv" "case_1" compile run export
   ```

   where 

   - "case1_searchSpace.csv" is the name of the csv file assumed to be located inside `comparing-tile-sizes`
   - "case 1" here is the name of desired output directory

### Shortcut commands

Only compile:

```
sh run_experiment.sh "tilingSchemes.csv" "outputFolderName" no compile no no
```

Check results of compilation (did any builds fail?)

```
sh run_experiment.sh "tilingSchemes.csv" "outputFolderName" no status no no
```

Only run:

```
sh run_experiment.sh "tilingSchemes.csv" "outputFolderName" no no run no
```

Only export results to a CSV:

```
sh run_experiment.sh "tilingSchemes.csv" "outputFolderName" no no no export
```

Updating to tile ANY of the kernels:

```
sh run_experiment.sh "case1_searchSpace.csv" case_1 genJsons no no export 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"

```

Regenerating csv for case 1:

```
sh run_experiment.sh "case1_searchSpace.csv" case_1 genJsons no no export 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```

Rgenerating csv for case 2:

```
sh run_experiment.sh "case2_searchSpace.csv" case2 genJsons no no export 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```

## dispatch 1 case 3:
only compile :
```
sh run_experiment.sh "dispatch_1_case3_searchSpace.csv" dispatch_1_case_3 genJsons compile no no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```
only check status of builds:
```
sh run_experiment.sh "dispatch_1_case3_searchSpace.csv" dispatch_1_case_3 no status no no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```
only run:
```
sh run_experiment.sh "dispatch_1_case3_searchSpace.csv" dispatch_1_case_3 no no run no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```
only export subset:
```
sh run_experiment.sh "dispatch_1_case3_searchSpace-export-early.csv" dispatch_1_case_3 no no no export 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```
## dispatch 7 cases 1 and 2
only create jsons:
```
sh run_experiment.sh "dispatch_7_case1_searchSpace.csv" dispatch_7_case_1 genJsons no no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
sh run_experiment.sh "dispatch_7_case2_searchSpace.csv" dispatch_7_case_2 genJsons no no no 2 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```
only compile:
```
time sh run_experiment.sh "dispatch_7_case1_searchSpace.csv" dispatch_7_case_1 no no run no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
time sh run_experiment.sh "dispatch_7_case2_searchSpace.csv" dispatch_7_case_2 no no run no 2 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```