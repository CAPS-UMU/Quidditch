echo "$1 is the searchspace, $2 is the folder to save output, "
echo "$3 $4 $5 $6 are the steps and $7 is the dispatchname"
echo "$8 says whether to skip email notification"
#. run_experiment.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" no no no export 1 "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
#. startNoHangUpRun.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" no no no export "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"
# nohup ./startNoHangUpRun.sh "1x600x600wm-n-k_case1_searchSpace.csv" "1x600x600wm-n-k" no no no export "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64"

cd "/home/hoppip/Quidditch/comparing-tile-sizes"
. run_experiment.sh $1 $2 $3 $4 $5 $6 1 $7
cd "/home/hoppip/Quidditch/"

if [[ "$8" == "skip" ]];
    then
    echo "skipping email notification"
    else
    pip install mechanize
    python notifyJobFinished.py "$7" "steps $3 $4 $5 $6 - finished"
fi

echo "JOB FINISHED"