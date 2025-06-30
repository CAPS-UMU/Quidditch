echo -e "run_linear_layer.sh: ATTN: Run this script INSIDE directory Quidditch/fakeNN/\n"
here=$(pwd) # save current directory so we can return to it
basename=`basename $1 | sed 's/[.][^.]*$//'` # strip search space argument of its .csv extension
searchSpaceCSV="$here/linear-layer-search-space/$basename.csv"
# constants derived from present working directory (here)
goldenOutputDir="$here/linear-layer-search-space/golden-out"
finalOutputDir="$here/linear-layer-search-space/out"
goldenJsonOutputDir="$here/linear-layer-search-space/golden-tiling-schemes"
jsonOutputDir="$here/linear-layer-search-space/tiling-schemes"
# constants derived ASSUMING present working directory = Quidditch/fakeNN
buildDir="../build"
fakeNNDir="../runtime/samples/fakeNN"
fakeNNExec="$buildDir/runtime/samples/fakeNN/FakeNN"
verilator="../toolchain/bin"
# constants derived from user input
genJsonsFlag=$2
compileFlag=$3
runFlag=$4
exportFlag=$5
echo -e "\nargs passed in are genJsons:$genJsonsFlag compile:$compileFlag run:$runFlag and export:$exportFlag :)"

# Example usage:
#. run_linear_layer.sh one-run.csv genJsons compile run export

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

#"main\$async_dispatch_0_matmul_transpose_b_2x120x40_f64"

## generate json files
if [[ "$genJsonsFlag" == "genJsons" ]];
    then
    echo -e "\nrun_linear_layer.sh: generating json files from the search space..."
    python generateTileSizeJSONFiles.py $searchSpaceCSV $jsonOutputDir $goldenJsonOutputDir
fi

## compile
if [[ "$compileFlag" == "compile" ]];
    then
    ## compile
    #sh compileGrapefruits.sh $1 $experimentName
    echo -e "\nrun_linear_layer.sh: build away!"
    . compile.sh $searchSpaceCSV $buildDir $goldenOutputDir $finalOutputDir $goldenJsonOutputDir $jsonOutputDir $fakeNNDir
    
    ## check compilation results
    else if [[ "$compileFlag" == "status" ]];
             then
             #sh compileGrapefruits.sh $1 $experimentName status
             echo -e "\nrun_linear_layer.sh: check status of builds instead of building!"
        fi
fi

## run
if [[ "$runFlag" == "run" ]];
    then
    echo -e "\nrun_linear_layer.sh: if golden hasn't been run, run golden"
    echo -e "\nrun_linear_layer.sh: if entry hasn't been run, run it"
    #sh runGrapefruits.sh $1 $experimentName
fi


## export 
if [[ "$exportFlag" == "correctness" ]];
    then
    echo -e "\nrun_linear_layer.sh: check run outputs for correctness"
    #sh scrutinizeGrapefruits.sh $1 $experimentName $7 $8
fi
if [[ "$exportFlag" == "export" ]];
    then
    echo -e "\nrun_linear_layer.sh: export results to a csv"
    #sh scrapeGrapefruits.sh $1 $experimentName $7 $8
fi


