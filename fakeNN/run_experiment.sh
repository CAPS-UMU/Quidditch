echo "run_experiment.sh: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/"
here=$(pwd) # save current directory so we can return to it
# script-specific constants
searchSpaceCSV="$here/$1"
experimentName="$2"
finalOutputDirectory="$here/$experimentName"
jsonOutputDirectory="$here/$experimentName/tile-sizes-to-test"
## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

## generate json files
if [[ "$3" == "genJsons" ]];
    then
    echo "run_experiment.sh: generating json files from the search space..."
    mkdir -p $jsonOutputDirectory
    python generateTileSizeJSONFiles.py $1 $8 $jsonOutputDirectory
fi

## compile
if [[ "$4" == "compile" ]];
    then
    ## compile
    sh compileGrapefruits.sh $1 $experimentName
    ## check compilation results
    else if [[ "$4" == "status" ]];
             then
             sh compileGrapefruits.sh $1 $experimentName status
        fi
fi

## run
if [[ "$5" == "run" ]];
    then
    sh runGrapefruits.sh $1 $experimentName
fi


## export 
if [[ "$6" == "correctness" ]];
    then
    sh scrutinizeGrapefruits.sh $1 $experimentName $7 $8
fi
if [[ "$6" == "export" ]];
    then
    sh scrapeGrapefruits.sh $1 $experimentName $7 $8
fi


