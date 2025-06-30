echo -e "\tcompile.sh: ATTN: NEVER run this script directly; Run script run_linear_layer.sh instead."
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
# constants defined by script, according to what I have in fakeNN right now
origCosts="fakeNN-tile-sizes-costs" # (.txt)
origScheme="fakeNN-tile-sizes"      # (.json)
origHeader="fakeNN_util"            # (.h)
# constant defined by script, according to the templates folder contents
cmakeEpilog="CMakeLists-epilog"
cHeaderProlog="fakeNN_util-prolog"
# echo -e "yohoho $1\n $2\n $3\n $4\n $5\n $6\n $7\n"
# echo "hoodle"
# ls "$2"
# echo "hoodle"

# echo "iree_turbine(SRC fakeNN.py DST \${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc DTYPE "f64" M "$M" N "$N" K "$K")"
# echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc DST fakeNN FLAGS --mlir-disable-threading --iree-quidditch-time-disp=fakenn"$M"x"$N"x"$K" --iree-quidditch-export-costs=$exportCosts --iree-quidditch-import-tiles=$importTiles)"
# echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc LLVM DST fakeNN_llvm FLAGS --iree-quidditch-time-disp=fakenn"$M"x"$N"x"$K" --iree-quidditch-export-costs=$exportCosts --iree-quidditch-import-tiles=$importTiles)"


# generate cmakelists.txt and C header file given 
# 1. the tile sizes json file basename
# 2. directory in which to save the cmakelists.txt file (do NOT use a relative path!)
# 3. tile size costs txt file basename
# 4. C header file basename
gen_cmakelists_and_source(){
    # echo ""
    echo "HELLO: tile scheme file is $1"
    echo "HELLO: directory from which to pull tile scheme from is $2"
    echo "HELLO: directory to save in is $3"
    jsons=$2
    out=$3
    ts=$1
    echo "HELLO: M=$4, N=$5, K=$6, m=$7, n=$8, k=$9."
    if [[ "$ts" == "original" ]]; 
    then 
        # restore set up for 2x120x40 with tile sizes 0-0-60
        cp "$templates/$origScheme.json" "$out/$origScheme.json" 
        cp "$templates/$origCosts.json" "$out/$origCosts.json" 
        cp "$templates/CMakeLists.txt" "$out/CMakeLists.txt"
        cp "$templates/$origHeader.h" "$out/$origHeader.h"
    else
        # copy custom tile sizes, costs, CMakeLists.txt and C header into destination
        importTiles="$out/$origScheme.json" 
        exportCosts="$out/$origCosts.txt"
        # cp source destination
        cp "$jsons/$ts.json" $importTiles           # copy tile sizes
        cp "$templates/$origCosts.txt" $exportCosts # copy tile costs (not really used)
        # create custom CMakeLists.txt
        echo "iree_turbine(SRC fakeNN.py DST \${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc DTYPE \"f64\" M "$M" N "$N" K "$K")" > "$out/CMakeLists.txt"
        echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc DST fakeNN FLAGS --mlir-disable-threading --iree-quidditch-time-disp=fakenn"$M"x"$N"x"$K" --iree-quidditch-export-costs=$exportCosts --iree-quidditch-import-tiles=$importTiles)" >> "$out/CMakeLists.txt"
        echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/fakeNN.mlirbc LLVM DST fakeNN_llvm FLAGS --iree-quidditch-time-disp=fakenn"$M"x"$N"x"$K" --iree-quidditch-export-costs=$exportCosts --iree-quidditch-import-tiles=$importTiles)" >> "$out/CMakeLists.txt"
        cat "$templates/$cmakeEpilog.txt" >> "$out/CMakeLists.txt"
        # create custom C header
        cat "$templates/$cHeaderProlog.h" > "$out/$origHeader.h"
        echo "#define mDim $M" >> "$out/$origHeader.h"
        echo "#define nDim $N" >> "$out/$origHeader.h"
        echo "#define kDim $K" >> "$out/$origHeader.h"
   fi
}

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo -e "\tERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

# # clear;sh scrapeGrapefruits.sh case1_searchSpace.csv "case_1"
# # clear; sh compileGrapefruits.sh case1_searchSpace.csv "case_1"
# # sh runGrapefruits.sh case1_searchSpace.csv "case_1"

# # existingExperiments=()
# # missingExperiments=()

# # for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
# #         do
# #         experimentResults="$here/$ts/run_output.txt"
# #         res=$(ls $experimentResults 2>/dev/null)
# #         if [[ $experimentResults == $res ]]; 
# #             then 
# #             existingExperiments+=("$experimentResults")
# #             else
# #             missingExperiments+=("$experimentResults")
# #         fi
# # done
# # echo "compileGrapefruits.sh: generating json files from the search space..."
# # python generateTileSizeJSONFiles.py $searchSpaceCSV

# # rm --f -R "$compileOutputDirectory" # delete previous outputs
# # mkdir "$compileOutputDirectory" # create fresh output folder

# if [[ "$3" == "status" ]];
#     then
#     ## check whether each build was successful
#     for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
#         do
#         basename=$ts # TODO: rename basname as ts everywhere
#         echo "checking $basename.json build..." # inform user we are checking build associated with $basename.json
#         grep "kernel does not fit into L1 memory and cannot be compiled" "$compileOutputDirectory/$basename/buildOutput.txt"
#         grep "Troublesome file path is" "$compileOutputDirectory/$basename/buildOutput.txt"
#         grep "FAILED: runtime-prefix/src/runtime-stamp/runtime-build" "$compileOutputDirectory/$basename/buildOutput.txt"
#         cd $here
#         done
#         gen_cmakelists "original" $grapefruitDir $exportedCostFile # generate original CMakeLists.txt
#     exit 0
# fi

#echo -e "\tcompile.sh: generating the cmake files and compiling..."
#echo -e "\tcompile.sh:processing each point in the search space $basename"
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
        golden=$(echo "$M""x""$N""x""$K""w0-0-0")
        # if build does not exist, create files and build
        buildOutputFile="$finalOutputDir/$ts/output.txt"
        res=$(ls $buildOutputFile 2>/dev/null)
        if [[ $buildOutputFile != $res ]]; 
        then 
            echo -e "\tcompile.sh: creating build for $ts"
            myBuildDir="$finalOutputDir/$ts"
            rm -r -f $myBuildDir 2> /dev/null
            mkdir $myBuildDir
            gen_cmakelists_and_source $ts $jsonOutputDir $myBuildDir $M $N $K $m $n $k
        else
            echo -e "\tcompile.sh: using cached build for $ts"
        fi
        # if golden build does not exist, create golden files and build
        goldenOutputFile="$goldenOutputDir/$golden/output.txt"
        res=$(ls $goldenOutputFile 2>/dev/null)
        if [[ $goldenOutputFile != $res ]]; 
        then 
            echo -e "\tcompile.sh: creating golden build for $golden"
            #gen_cmakelists_and_source $golden $jsonOutputDir $fakeNNDir $M $N $K 0 0 0
        else
            echo -e "\tcompile.sh: using cached golden  build for $golden"
        fi
        # echo -e "\t$M"
        # echo -e "\t$N"
        # echo -e "\t$K"
        # echo -e "\t$m"
        # echo -e "\t$n"
        # echo -e "\t$k"
        # dispatchNameTemplate="main\$async_dispatch_0_matmul_transpose_b_MxNxK_f64"
        # dispatchName="${dispatchNameTemplate/MxNxK/"$M"x"$N"x"$K"}"
        # echo -e "\tso the dispatch name is $dispatchName"

done




    # for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
    #     do
    #     basename=$ts # TODO: rename basname as ts everywhere
    #     mkdir -p "$compileOutputDirectory/$basename" # create a local subfolder for this set of tile sizes
    #     echo "$basename.json" # inform user we are about to start processing $basename.json
    #     exportedCostFile="$compileOutputDirectory/$basename/tilingCosts.json" # using full path here
    #     gen_cmakelists "$ts.json" $grapefruitDir $exportedCostFile # generate basename-specific CMakeLists.txt
    #     gen_cmakelists "$ts.json" "$compileOutputDirectory/$basename" $exportedCostFile # save a copy of it in our local subfolder
    #     cd $buildDir
    #     cmake .. -GNinja \
    #     -DCMAKE_C_COMPILER=clang \
    #     -DCMAKE_CXX_COMPILER=clang++ \
    #     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    #     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    #     -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &> "$compileOutputDirectory/$basename/cmakeOutput.txt"
    #     ninja -j 20 &> "$compileOutputDirectory/$basename/buildOutput.txt"
    #     grep "kernel does not fit into L1 memory and cannot be compiled" "$compileOutputDirectory/$basename/buildOutput.txt"
    #     grep "Troublesome file path is" "$compileOutputDirectory/$basename/buildOutput.txt"
    #     cd $here
    #     # copy generated executable to local folder
    #     cp $grapefruitExec "$compileOutputDirectory/$basename/GrapeFruit" # copy SRC to DST  
    #     done
    #     gen_cmakelists "original" $grapefruitDir $exportedCostFile # generate original CMakeLists.txt
       
