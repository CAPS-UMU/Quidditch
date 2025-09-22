echo "scrapeGrapefruits.sh: ATTN: NEVER run this script directly; instead, call it from run_experiment.sh"
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
# build-specific constants
grapefruitDir="$quidditchDir/runtime/samples/grapeFruit"
buildDir="$quidditchDir/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="$quidditchDir/toolchain/bin"

## helper function
parse_exp_result(){
    filePath=$1
    dispatchNo=$2 # legacy value
    dispatchName=$3
    #echo "HOLAAAAA filePath is $filePath, dispatchNo is $dispatchNo and dispatchName is $dispatchName"
    #ts=$(python3 parseTSFromPath.py $filePath)
    dispatchNo=$(python3 parseDispatchNo.py $dispatchName)
    basename=`basename $(echo $filePath | sed 's/run_output.txt//') | sed 's/[.][^.]*$//'`
    kernelTime=$(grep -E "^(dispatch) $dispatchNo: ([0-9]*) - ([0-9]*) = ([0-9]*)" "$filePath" | grep -oE '[^[:space:]]+$')
    totalTime=$(grep -E "cycles ([0-9]*)" "$filePath" | grep -oE '[^[:space:]]+$')
    echo "$basename,$dispatchName,$kernelTime,$totalTime" >> $parsedResultsCSV
}

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

existingExperiments=()
missingExperiments=()

## scrape experiments from the search space
#for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
eatNum='^([0-9])([0-9])*'
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
        basename=$(echo "$m""-""$n""-""$k")
        experimentResults="$here/$scrapeName/$basename/run_output.txt"
        res=$(ls $experimentResults 2>/dev/null)
        if [[ $experimentResults == $res ]]; 
            then 
            existingExperiments+=("$experimentResults")
            else
            missingExperiments+=("$experimentResults")
        fi
done


echo "scrapeGrapefruits.sh: we will skip the following missing experiments:"
for element in "${missingExperiments[@]}"
do
    echo $element
done
echo "scrapeGrapefruits.sh: we will export the following experiment results to a csv:"
for element in "${existingExperiments[@]}"
do
    echo $element
done

## generate fresh CSV output file
rm "$parsedResultsCSV"
rm "$here/$scrapeName/$scrapeName-graphing.csv"
# rmdir "$here/$scrapeName" 
# mkdir "$here/$scrapeName"
touch "$parsedResultsCSV"
echo "JSON Name,Kernel Name,Kernel Time,Total Time" > $parsedResultsCSV #$results
#echo "JSON Name,Kernel Name,Kernel Time,Total Time" >> $parsedResultsCSV

## parse each experiment's run_output.txt
## and append the parsed info to the CSV file
for element in "${existingExperiments[@]}"
do  
    parse_exp_result $element $3 $4
done

## merge search space info with parsed results for graphing
python merge.py $searchSpaceCSV $parsedResultsCSV "JSON Name"
cp "merged.csv" "$here/$scrapeName/$scrapeName-graphing.csv"
rm "merged.csv"





# notes below!
# some helpful grep patterns to remember, 
# even if not all of them used in current script:
#  echo "third grep"
#     grep -E "(:alpha:|[0-9]*) = [0-9]*" "$filePath"
#     echo "fourth grep"
#     grep -E "^(dispatch) ([0-9]*): ([0-9]*) - ([0-9]*) = ([0-9]*)" "$filePath"
#     echo "fifth grep"
#     grep -E "cycles ([0-9]*)" "$filePath"