echo -e "\texport.sh: ATTN: NEVER run this script directly; Run script run_linear_layer.sh instead."
# constants derived from present working directory (here)
here=$(pwd) # save current directory so we can return to it
# constants derived from user input
searchSpaceCSV=$1
buildDir=$2
goldenOutputDir=$3
finalOutputDir=$4
goldenJsonOutputDir=$5
jsonOutputDir=$6
fakeNNDir=$7
templates=$8
fakeNNExec=$9
correctness=${10}
# constants defined by script, according to what I have in fakeNN right now
origCosts="fakeNN-tile-sizes-costs" # (.txt)
origScheme="fakeNN-tile-sizes"      # (.json)
origHeader="fakeNN_util"            # (.h)
# constants defined by script, according to the templates folder contents
cmakeEpilog="CMakeLists-epilog"
cHeaderProlog="fakeNN_util-prolog"

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo -e "\tERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

## helper function
check_exp_result(){
    filePath=$1
    runOutputJustValues="./temp.txt"
    lineCount=$(wc --lines $filePath | head -n1 | sed -e 's/\s.*$//')
    theTail=$(($lineCount - 1))
    theHead=$(($lineCount - 19))
    tail -n $theTail $filePath | head -n $theHead > $runOutputJustValues
    goldPath=$2
    goldOutputJustValues="./temp2.txt"
    lineCount=$(wc --lines $goldPath | head -n1 | sed -e 's/\s.*$//')
    theTail=$(($lineCount - 1))
    theHead=$(($lineCount - 19))
    tail -n $theTail $goldPath | head -n $theHead > $goldOutputJustValues
    diffResult=$(diff $goldOutputJustValues $runOutputJustValues)
    if [[ $diffResult != "" ]]; 
        then 
        echo -e "\tERROR: $filePath contains incorrect results!"
        echo $diffResult
        else
        echo -e "\t$filePath OK"
    fi
    rm -f $goldOutputJustValues 2> /dev/null
    rm -f $runOutputJustValues 2> /dev/null
}

uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
eatNum='^([0-9])([0-9])*'
# check build status of entry requested by CSV
if [[ "$correctness" == "correctness" ]];
    then
        # check whether each run was successful
        for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
            do
            eatNum='^([0-9])([0-9])*'
            M=$(echo $ts | grep -oE $eatNum)
            tail=${ts#*x}
            N=$(echo $tail | grep -oE $eatNum)
            tail=${tail#*x}
            K=$(echo $tail | grep -oE $eatNum)
            tail=${tail#*w}
            m=$(echo $tail | grep -oE $eatNum)
            tail=${tail#*-}
            n=$(echo $tail | grep -oE $eatNum)
            tail=${tail#*-}
            k=$(echo $tail | grep -oE $eatNum)
            golden=$(echo "$M""x""$N""x""$K""w0-0-0")
            #echo -e "\texport.sh: checking $golden vs $ts..."
            output="$finalOutputDir/$ts/run_output.txt"
            goldenOutput="$goldenOutputDir/$golden/run_output.txt"
            check_exp_result $output $goldenOutput
        done
    else
        # export each entry requested by CSV
        missingExperiments=()
        # strip search space argument of its .csv extension
        basename=`basename $searchSpaceCSV | sed 's/[.][^.]*$//'`
        # generate fresh CSV output file
        results="$finalOutputDir/$basename-results.csv"
        touch $results
        echo "JSON Name,Kernel Name,Kernel Time,Total Time,M,N,K,m,n,k" > $results
        for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
                do
                eatNum='^([0-9])([0-9])*'
                M=$(echo $ts | grep -oE $eatNum)
                tail=${ts#*x}
                N=$(echo $tail | grep -oE $eatNum)
                tail=${tail#*x}
                K=$(echo $tail | grep -oE $eatNum)
                tail=${tail#*w}
                m=$(echo $tail | grep -oE $eatNum)
                tail=${tail#*-}
                n=$(echo $tail | grep -oE $eatNum)
                tail=${tail#*-}
                k=$(echo $tail | grep -oE $eatNum)
                dispatchNameTemplate="main\$async_dispatch_0_matmul_transpose_b_MxNxK_f64"
                dispatchName="${dispatchNameTemplate/MxNxK/"$M"x"$N"x"$K"}"
                # tiled results
                experimentResults="$finalOutputDir/$ts/run_output.txt"
                res=$(ls $experimentResults 2>/dev/null)
                if [[ $experimentResults == $res ]]; 
                    then 
                    echo -e "\texport results related to $ts"
                    kernelTime=$(grep -E "^(dispatch) 0: ([0-9]*) - ([0-9]*) = ([0-9]*)" "$experimentResults" | grep -oE '[^[:space:]]+$')
                    totalTime=$(grep -E "cycles ([0-9]*)" "$experimentResults" | grep -oE '[^[:space:]]+$')
                    echo "$ts,$dispatchName,$kernelTime,$totalTime,$M,$N,$K,$m,$n,$k" >> $results
                    else
                    missingExperiments+=("$experimentResults")
                fi
                # golden results
                golden=$(echo "$M""x""$N""x""$K""w0-0-0")
                experimentResults="$goldenOutputDir/$golden/run_output.txt"
                res=$(ls $experimentResults 2>/dev/null)
                if [[ $experimentResults == $res ]]; 
                    then 
                    echo -e "\texport results related to $golden"
                    kernelTime=$(grep -E "^(dispatch) 0: ([0-9]*) - ([0-9]*) = ([0-9]*)" "$experimentResults" | grep -oE '[^[:space:]]+$')
                    totalTime=$(grep -E "cycles ([0-9]*)" "$experimentResults" | grep -oE '[^[:space:]]+$')
                    echo "$golden,$dispatchName,$kernelTime,$totalTime,$M,$N,$K,0,0,0" >> $results
                    else
                    missingExperiments+=("$experimentResults")
                fi
        done
        echo "we had to skip the following missing experiments:"
        for element in "${missingExperiments[@]}"
        do
            echo $element
        done
        
fi
