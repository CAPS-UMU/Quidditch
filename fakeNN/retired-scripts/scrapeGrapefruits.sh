echo "scrapeGrapefruits.sh: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/"
here=$(pwd) # save current directory so we can return to it
# script-specific constants
tileSizes="/home/hoppip/Quidditch/comparing-tile-sizes/tile-sizes-to-test/*.json"
prologueFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-prologue.txt"
middleFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-middle-original.txt"
epilogueFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-epilogue.txt"
scrapeName="$2"
parsedResultsCSV="$here/$scrapeName/csv_experiment_results.csv"
searchSpaceCSV="$here/$1"
# build-specific constants
grapefruitDir="/home/hoppip/Quidditch/runtime/samples/grapeFruit"
buildDir="/home/hoppip/Quidditch/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="/home/hoppip/Quidditch/toolchain/bin"

## debugging
function_name(){
    echo "yohoho $1"
}

## helper function
parse_exp_result(){
    filePath=$1
    dispatchNo=$2
    dispatchName=$3
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

## scrape every experiment ever done
# for ts in $tileSizes
#         do
#         basename=`basename $ts | sed 's/[.][^.]*$//'` # note that $ts is a full file path
#         experimentResults="$here/$basename/run_output.txt"
#         res=$(ls $experimentResults 2>/dev/null)
#         if [[ $experimentResults == $res ]]; 
#             then 
#             existingExperiments+=("$experimentResults")
#             else
#             missingExperiments+=("$experimentResults")
#         fi
# done

## scrape experiments from the search space
for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
        do
        experimentResults="$here/$scrapeName/$ts/run_output.txt"
        res=$(ls $experimentResults 2>/dev/null)
        if [[ $experimentResults == $res ]]; 
            then 
            existingExperiments+=("$experimentResults")
            else
            missingExperiments+=("$experimentResults")
        fi
done


echo "we will skip the following missing experiments:"
for element in "${missingExperiments[@]}"
do
    echo $element
done
echo "we will export the following experiment results to a csv:"
for element in "${existingExperiments[@]}"
do
    echo $element
done

## generate fresh CSV output file
rm "$parsedResultsCSV"
rm "$here/$scrapeName/graphing.csv"
# rmdir "$here/$scrapeName" 
# mkdir "$here/$scrapeName"
touch "$parsedResultsCSV"
echo "JSON Name,Kernel Name,Kernel Time,Total Time" >> $parsedResultsCSV

## parse each experiment's run_output.txt
## and append the parsed info to the CSV file
for element in "${existingExperiments[@]}"
do
    parse_exp_result $element $3 $4
done

## merge search space info with parsed results for graphing
python merge.py $searchSpaceCSV $parsedResultsCSV "JSON Name"
cp "merged.csv" "$here/$scrapeName/graphing.csv"
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