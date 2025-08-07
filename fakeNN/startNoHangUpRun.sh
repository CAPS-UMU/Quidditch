echo "$1 is the searchspace"
echo "$2 $3 $4 $5 are the steps"
echo "$6 says whether to skip email notification"

# example run
# nohup bash startNoHangUpRun.sh one-more-run.csv genJsons no no no skip &> one-more-run.output &
# nohup bash startNoHangUpRun.sh one-more-run.csv genJsons compile run no &> one-more-run.output &

. run_linear_layer.sh $1 $2 $3 $4 $5

if [[ "$8" == "skip" ]];
    then
    echo "skipping email notification"
    else
    pip install mechanize
    python notifyJobFinished.py "$1" "steps $2 $3 $4 $5 - finished"
fi

echo "JOB FINISHED"