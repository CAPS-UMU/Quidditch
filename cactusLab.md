# Cactus Lab
How to log into my lab computer with ssh, launch jobs, and log out of ssh without killing those jobs.
0. Make sure you have your `venv` enabled beforehand!! `source venv/bin/activate`
1. Use vscode remote explorer to ssh into my `Quidditch` repo
2. Pick the dispatch I want to time. For example, for dispatch 8, I need to specify
    - dispatch tile search space: `1x600x600wm-n-k_case1_searchSpace.csv`
    - output folder: `1x600x600wm-n-k`
    - dispatch name as an escaped string: `"main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"`
    - unique job output file name: `myJobOutputD8.out`
3. Use `nohup` to launch the job.
     ```
    nohup bash startNoHangUpRun.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" no no no export "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" > myJobOutputD8.out &
    ```
4. wait for an email :)
## Example Launches
1. genJsons
    ```
    nohup bash startNoHangUpRun.sh "1x600x400wm-n-k_case1_searchSpace_only_one.csv" "1x600x600wm-n-k_only_one" genJsons no no no "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" > myJobOutputD7.out &
    ```
2. compile
    ```
    nohup bash startNoHangUpRun.sh "1x600x400wm-n-k_case1_searchSpace_only_one.csv" "1x600x600wm-n-k_only_one" no compile no no "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" > myJobOutputD7.out &
    ```
3. status
    ```
    nohup bash startNoHangUpRun.sh "1x600x400wm-n-k_case1_searchSpace_only_one.csv" "1x600x600wm-n-k_only_one" no status no no "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" > myJobOutputD7.out &
    ```
4. run
    ```
    nohup bash startNoHangUpRun.sh "1x600x400wm-n-k_case1_searchSpace_only_one.csv" "1x600x600wm-n-k_only_one" no no run no "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" > myJobOutputD7.out &
    ```
5. correctness
    ```
    hoodle
    ```
6. export
    ```
    nohup bash startNoHangUpRun.sh "1x600x400wm-n-k_case1_searchSpace_only_one.csv" "1x600x600wm-n-k_only_one" no no no export "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" > myJobOutputD7.out &
    ```
## Commands Run for NsNet2 with correctness checked...
```
nohup bash startNoHangUpRun.sh "1x600x400wm-n-k_case1_searchSpace.csv" "1x600x400wm-n-k" genJsons no no no "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" > myJobOutputD7.out &

nohup bash startNoHangUpRun.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" genJsons no no no "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" > myJobOutputD8.out &

nohup bash startNoHangUpRun.sh "1x1200x400wm-n-k_case1_searchSpace.csv" "1x1200x400wm-n-k" genJsons no no no "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64" > myJobOutputD1.out &

nohup bash startNoHangUpRun.sh "1x400x161wm-n-k_case1_searchSpace.csv" "1x400x161wm-n-k" genJsons no no no "main\$async_dispatch_0_matmul_transpose_b_1x400x161_f64" > myJobOutputD0.out &
```