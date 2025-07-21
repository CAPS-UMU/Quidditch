# making search spaces match

Reference command:

```
sh run_experiment.sh "case2_searchSpace.csv" case2 genJsons compile run export 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
```

notes

```
searchSpaceCSV="$here/$1"
experimentName="$2"
finalOutputDirectory="$here/$experimentName"
jsonOutputDirectory="$here/$experimentName/tile-sizes-to-test"
```

I need to time 4 nsnet kernels...

```
comparing-tile-sizes/1x400x161wm-n-k_case1_searchSpace.csv

comparing-tile-sizes/1x600x400wm-n-k_case1_searchSpace.csv
comparing-tile-sizes/1x600x600wm-n-k_case1_searchSpace.csv
comparing-tile-sizes/1x1200x400wm-n-k_case1_searchSpace.csv
```

dispatch name cheatsheet:

```
main$async_dispatch_9_matmul_transpose_b_1x161x600_f64 : 

main$async_dispatch_8_matmul_transpose_b_1x600x600_f64 : 1080000 15

main$async_dispatch_0_matmul_transpose_b_1x400x161_f64 : 193200 10

main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64 : 1440000 30

main$async_dispatch_7_matmul_transpose_b_1x600x400_f64 : 720000 15
```

1. generate jsons
   ```
   . run_experiment.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" genJsons no no no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
   
   . run_experiment.sh "1x400x161wm-n-k_case1_searchSpace.csv" "1x400x161wm-n-k" genJsons no no no 1 "main\$async_dispatch_0_matmul_transpose_b_1x400x161_f64"
   
   . run_experiment.sh "1x1200x400wm-n-k_case1_searchSpace.csv" "1x1200x400wm-n-k" genJsons no no no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
   
   . run_experiment.sh "1x600x400wm-n-k_case1_searchSpace.csv" "1x600x400wm-n-k" genJsons no no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
   ```

2. compile
   ```
   . run_experiment.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" no compile no no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
   
   . run_experiment.sh "1x400x161wm-n-k_case1_searchSpace.csv" "1x400x161wm-n-k" no compile no no 1 "main\$async_dispatch_0_matmul_transpose_b_1x400x161_f64"
   
   . run_experiment.sh "1x1200x400wm-n-k_case1_searchSpace.csv" "1x1200x400wm-n-k" no compile no no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
   
   . run_experiment.sh "1x600x400wm-n-k_case1_searchSpace.csv" "1x600x400wm-n-k" no compile no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
   ```

3. status
   ```
   . run_experiment.sh "1x1200x400wm-n-k_case1_searchSpace.csv" "1x1200x400wm-n-k" no status no no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
   ```

   

4. run

   ```
   . run_experiment.sh "1x1200x400wm-n-k_case1_searchSpace.csv" "1x1200x400wm-n-k" no no run no 1 "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"
   
   . run_experiment.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" no no run no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
   ```


## Debugging

generate jsons

```
. run_experiment.sh "1x600x600wm-n-k_case1_searchSpace-debug.csv" "1x600x600wm-n-k" genJsons no no no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
. run_experiment.sh "1x600x400wm-n-k_case1_searchSpace-debug.csv" "1x600x400wm-n-k" genJsons no no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

compile

```
. run_experiment.sh "1x600x600wm-n-k_case1_searchSpace-debug.csv" "1x600x600wm-n-k" no compile no no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
. run_experiment.sh "1x600x400wm-n-k_case1_searchSpace-debug.csv" "1x600x400wm-n-k" no compile no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

run
```
. run_experiment.sh "1x600x600wm-n-k_case1_searchSpace-debug.csv" "1x600x600wm-n-k" no no run no 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
```



## segfaults

```
starting new batch...
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-8/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-10/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-16/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-20/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-25/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-40/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-50/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-80/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-40-100/GrapeFruit
/home/hoppip/Quidditch/comparing-tile-sizes/1x600x400wm-n-k/0-120-8/GrapeFruit
[1]   Segmentation fault      (core dumped) nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[2]   Done                    nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[3]   Done                    nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[4]   Done                    nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[5]   Segmentation fault      (core dumped) nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[6]   Segmentation fault      (core dumped) nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[7]   Segmentation fault      (core dumped) nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[8]   Done                    nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[9]-  Segmentation fault      (core dumped) nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)
[10]+  Done                    nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt"  (wd: ~/Quidditch/toolchain/bin)
(wd now: ~/Quidditch/comparing-tile-sizes)

```

9,7,6,5

## to run

generate json files:

```
. run_experiment.sh "1x600x400wm-n-k_case1_searchSpace.csv" "1x600x400wm-n-k" genJsons no no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

compile for each json file:

```
. run_experiment.sh "1x600x400wm-n-k_case1_searchSpace.csv" "1x600x400wm-n-k" no compile no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

trying mini search space first...

compile

```
. run_experiment.sh "mini.csv" "1x600x400wm-n-k" no compile no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

status

```
. run_experiment.sh "mini.csv" "1x600x400wm-n-k" no status no no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

run

```
. run_experiment.sh "mini.csv" "1x600x400wm-n-k" no no run no 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```

export

```
. run_experiment.sh "mini.csv" "1x600x400wm-n-k" no no no export 1 "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64"
```



correctness

export

