echo -e "run_linear_layer.sh: ATTN: Run this script INSIDE directory Quidditch/fakeNN/"
here=$(pwd) # save current directory so we can return to it
basename=`basename $1 | sed 's/[.][^.]*$//'` # strip search space argument of its .csv extension
searchSpaceCSV="$here/linear-layer-search-space/$basename.csv"
# constants derived from present working directory (here)
goldenOutputDir="$here/linear-layer-search-space/golden-out"
finalOutputDir="$here/linear-layer-search-space/out"
goldenJsonOutputDir="$here/linear-layer-search-space/golden-tiling-schemes"
jsonOutputDir="$here/linear-layer-search-space/tiling-schemes"
templates="$here/linear-layer-search-space/templates"
# constants with COMPLETE PATH SPECIFIC TO MY COMPUTER
quidditchDir="/home/hoppip/Quidditch" # <------------ modify this path with path to YOUR quidditch directory!
# constants with derived from this COMPLETE PATH SPECIFIC TO MY COMPUTER
buildDir="$quidditchDir/build"
fakeNNDir="$quidditchDir/runtime/samples/fakeNN"
fakeNNExec="$buildDir/runtime/samples/fakeNN/FakeNN"
verilator="$quidditchDir/toolchain/bin"
# constants derived from user input
genJsonsFlag=$2
compileFlag=$3
runFlag=$4
exportFlag=$5
echo -e "args passed in are genJsons:$genJsonsFlag compile:$compileFlag run:$runFlag and export:$exportFlag :)"

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
    echo -e "\nrun_linear_layer.sh: Generate JSON files..."
    python generateTileSizeJSONFiles.py $searchSpaceCSV $jsonOutputDir $goldenJsonOutputDir
fi

## compile
if [[ "$compileFlag" == "compile" ]];
    then
    ## compile
    #sh compileGrapefruits.sh $1 $experimentName
    echo -e "\nrun_linear_layer.sh: build away!"
    . compile.sh $searchSpaceCSV $buildDir $goldenOutputDir $finalOutputDir $goldenJsonOutputDir $jsonOutputDir $fakeNNDir $templates $fakeNNExec
    
    ## check compilation results
    else if [[ "$compileFlag" == "status" ]];
             then
             echo -e "\nrun_linear_layer.sh: CHECK status of builds..."
             . compile.sh $searchSpaceCSV $buildDir $goldenOutputDir $finalOutputDir $goldenJsonOutputDir $jsonOutputDir $fakeNNDir $templates $fakeNNExec "status"
   
        fi
fi

## run
if [[ "$runFlag" == "run" ]];
    then
    echo -e "\nrun_linear_layer.sh: RUN each csv entry..."
    . run.sh $searchSpaceCSV $buildDir $goldenOutputDir $finalOutputDir $goldenJsonOutputDir $jsonOutputDir $fakeNNDir $templates $fakeNNExec $verilator
fi


## export 
if [[ "$exportFlag" == "correctness" ]];
    then
    echo -e "\nrun_linear_layer.sh: Check for CORRECTNESS..."
    . export.sh $searchSpaceCSV $buildDir $goldenOutputDir $finalOutputDir $goldenJsonOutputDir $jsonOutputDir $fakeNNDir $templates $fakeNNExec "correctness"
fi
if [[ "$exportFlag" == "export" ]];
    then
    echo -e "\nrun_linear_layer.sh: EXPORT results to a csv..."
    . export.sh $searchSpaceCSV $buildDir $goldenOutputDir $finalOutputDir $goldenJsonOutputDir $jsonOutputDir $fakeNNDir $templates $fakeNNExec "export"
fi


