echo "scrutinizeGrapefruits.sh: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/"
here=$(pwd) # save current directory so we can return to it
# script-specific constants
quidditchDir="/home/hoppip/Quidditch"
tileSizes="$quidditchDir/comparing-tile-sizes/tile-sizes-to-test/*.json"
prologueFile="$quidditchDir/comparing-tile-sizes/cmakelist-prologue.txt"
middleFile="$quidditchDir/comparing-tile-sizes/cmakelist-middle-original.txt"
epilogueFile="$quidditchDir/comparing-tile-sizes/cmakelist-epilogue.txt"
scrapeName="$2"
parsedResultsCSV="$here/$scrapeName/csv_experiment_results.csv"
searchSpaceCSV="$here/$1"
goldenOutputFile="$quidditchDir/comparing-tile-sizes/nsnet2_golden_output.txt"
# build-specific constants
grapefruitDir="$quidditchDir/runtime/samples/grapeFruit"
buildDir="$quidditchDir/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="$quidditchDir/toolchain/bin"

## helper function
compare_result(){
    filePath=$1
    runOutputJustValues="$2/temp.txt"
    rm $runOutputJustValues 2> /dev/null
    lineCount=$(wc --lines $filePath | head -n1 | sed -e 's/\s.*$//')
    theTail=$(($lineCount - 1))
    tail -n $theTail $filePath | head -n 161 > $runOutputJustValues
    diffResult=$(diff $goldenOutputFile $runOutputJustValues)
    if [[ $diffResult != "" ]]; 
        then 
        echo "ERROR: $filePath contains incorrect results!"
        cp $filePath "$2/run_output_WRONG.txt"
        rm -rf $filePath
        echo $diffResult
        else
        echo "$filePath OK"
        fi
}

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

missingExperiments=()

echo "checking..."

## scrape experiments from the search space
for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
        do
        experimentResults="$here/$scrapeName/$ts/run_output.txt"
        tempOutput="$here/$scrapeName/$ts"
        res=$(ls $experimentResults 2>/dev/null)
        if [[ $experimentResults == $res ]]; 
            then 
            compare_result "$experimentResults" "$tempOutput"
            else
            missingExperiments+=("$experimentResults")
        fi
done


echo "we skipped the following missing experiments:"
for element in "${missingExperiments[@]}"
do
    echo $element
done
