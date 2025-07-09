echo -e "compileGrapefruits.sh: ATTN: This should only be run by run_experiment.sh"
here=$(pwd) # save current directory so we can return to it
# script-specific constants"
quidditchDir="/home/emily/Quidditch"
tileSizes="$quidditchDir/comparing-tile-sizes/tile-sizes-to-test/*.json"
prologueFile="$quidditchDir/comparing-tile-sizes/cmakelist-prologue.txt"
middleFile="$quidditchDir/comparing-tile-sizes/cmakelist-middle-original.txt"
epilogueFile="$quidditchDir/comparing-tile-sizes/cmakelist-epilogue.txt"
searchSpaceCSV="$here/$1"
echo "searchSpaceCSV is $searchSpaceCSV"
compileOutputDirectory="$here/$2"
tileSizesToTest="$here/$2/tile-sizes-to-test"
# build-specific constants
grapefruitDir="$quidditchDir/runtime/samples/grapeFruit"
buildDir="$quidditchDir/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="$quidditchDir/toolchain/bin"

# debugging
function_name(){
    echo "yohoho $1 $2"
}

# generate cmakelists.txt file given 
# 1. the tile sizes json (as form basename.json)
# 2. directory in which to save the cmakelists.txt file (do NOT use a relative path!)
gen_cmakelists(){
    # echo ""
    tile="$tileSizesToTest/$1"
    # echo "directory to save in is $2"
    if [[ "$1" == "original" ]]; 
    then 
       cat $prologueFile > "$2/CMakeLists.txt"
       cat $middleFile >> "$2/CMakeLists.txt"
       cat $epilogueFile >> "$2/CMakeLists.txt"
    else
        cat $prologueFile > "$2/CMakeLists.txt"
        echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc DST grapeFruit FLAGS --mlir-disable-threading --iree-quidditch-time-disp=grapeFruit --iree-quidditch-import-tiles=$tile)" >> "$2/CMakeLists.txt"
        echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc LLVM DST grapeFruit_llvm FLAGS --mlir-disable-threading --iree-quidditch-time-disp=grapeFruit --iree-quidditch-import-tiles=$tile)" >> "$2/CMakeLists.txt"
        cat $epilogueFile >> "$2/CMakeLists.txt"
   fi
}

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi


if [[ "$3" == "status" ]];
    then
    ## check whether each build was successful
    for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
        do
        basename=$ts # TODO: rename basname as ts everywhere
        echo "checking $basename.json build..." # inform user we are checking build associated with $basename.json
        grep "kernel does not fit into L1 memory and cannot be compiled" "$compileOutputDirectory/$basename/buildOutput.txt"
        grep "Troublesome file path is" "$compileOutputDirectory/$basename/buildOutput.txt"
        grep "FAILED: runtime-prefix/src/runtime-stamp/runtime-build" "$compileOutputDirectory/$basename/buildOutput.txt"
        cd $here
        done
    gen_cmakelists "original" $grapefruitDir # generate original CMakeLists.txt
    else
        echo "compileGrapefruits.sh: generating the cmake files and compiling..."
        for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
            do
            basename=$ts # TODO: rename basname as ts everywhere
            mkdir -p "$compileOutputDirectory/$basename" # create a local subfolder for this set of tile sizes
            echo "$basename.json" # inform user we are about to start processing $basename.json
            #exportedCostFile="$compileOutputDirectory/$basename/tilingCosts.json" # using full path here
            gen_cmakelists "$ts.json" $grapefruitDir # generate basename-specific CMakeLists.txt
            gen_cmakelists "$ts.json" "$compileOutputDirectory/$basename" # save a copy of it in our local subfolder
            cd $buildDir
            cmake .. -GNinja \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &> "$compileOutputDirectory/$basename/cmakeOutput.txt"
            ninja -j 20 &> "$compileOutputDirectory/$basename/buildOutput.txt"
            grep "kernel does not fit into L1 memory and cannot be compiled" "$compileOutputDirectory/$basename/buildOutput.txt"
            grep "Troublesome file path is" "$compileOutputDirectory/$basename/buildOutput.txt"
            cd $here
            # copy generated executable to local folder
            cp $grapefruitExec "$compileOutputDirectory/$basename/GrapeFruit" # copy SRC to DST  
            done
        gen_cmakelists "original" $grapefruitDir # generate original CMakeLists.txt

fi






# old notes below...

# clear;sh scrapeGrapefruits.sh case1_searchSpace.csv "case_1"
# clear; sh compileGrapefruits.sh case1_searchSpace.csv "case_1"
# sh runGrapefruits.sh case1_searchSpace.csv "case_1"

# existingExperiments=()
# missingExperiments=()

# for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
#         do
#         experimentResults="$here/$ts/run_output.txt"
#         res=$(ls $experimentResults 2>/dev/null)
#         if [[ $experimentResults == $res ]]; 
#             then 
#             existingExperiments+=("$experimentResults")
#             else
#             missingExperiments+=("$experimentResults")
#         fi
# done
# echo "compileGrapefruits.sh: generating json files from the search space..."
# python generateTileSizeJSONFiles.py $searchSpaceCSV

# rm --f -R "$compileOutputDirectory" # delete previous outputs
# mkdir "$compileOutputDirectory" # create fresh output folder