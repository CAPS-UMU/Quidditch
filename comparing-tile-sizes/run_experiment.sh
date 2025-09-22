echo "run_experiment.sh: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/"
here=$(pwd) # save current directory so we can return to it
# script-specific constants
searchSpaceCSV="$here/$1"
experimentName="$2"
finalOutputDirectory="$here/$experimentName"
jsonOutputDirectory="$here/$experimentName/tile-sizes-to-test"
# constants derived from user input
genJsonsFlag=$3
compileFlag=$4
runFlag=$5
exportFlag=$6
echo -e "args passed in are genJsons:$genJsonsFlag compile:$compileFlag run:$runFlag and export:$exportFlag :)"

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"

fi

## generate json files
if [[ $genJsonsFlag == "genJsons" ]];
    then
    echo "run_experiment.sh: generating json files from the search space..."
    mkdir -p $jsonOutputDirectory
    python generateTileSizeJSONFiles.py $1 $8 $jsonOutputDirectory
fi

## compile
if [[ $compileFlag == "compile" ]];
    then
    ## compile
    . compileGrapefruits.sh $1 $experimentName
    ## check compilation results
    else if [[ $compileFlag == "status" ]];
             then
             . compileGrapefruits.sh $1 $experimentName "status"
        fi
fi

## run
if [[ $runFlag == "run" ]];
    then
    . runGrapefruits.sh $1 $experimentName
fi


## export 
if [[ $exportFlag == "correctness" ]];
    then
    . scrutinizeGrapefruits.sh $1 $experimentName $7 $8
fi
if [[ $exportFlag == "export" ]];
    then
    . scrapeGrapefruits.sh $1 $experimentName 1 $8 # 1 is a legacy argument; someday remove
fi


