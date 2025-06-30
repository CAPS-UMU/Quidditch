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

# echo -e "yohoho $1\n $2\n $3\n $4\n $5\n $6\n $7\n"
# echo "hoodle"
# ls "$2"
# echo "hoodle"



# generate cmakelists.txt file given 
# 1. the tile sizes json (as form basename.json)
# 2. directory in which to save the cmakelists.txt file (do NOT use a relative path!)
# 3. filepath to save the tile size costs (do NOT use a relative path!)
# gen_cmakelists(){
#     # echo ""
#     # echo "tile sizes file is $1"
#     tile="$tileSizesToTest/$1"
#     # echo "directory to save in is $2"
#     if [[ "$1" == "original" ]]; 
#     then 
#        cat $prologueFile > "$2/CMakeLists.txt"
#        cat $middleFile >> "$2/CMakeLists.txt"
#        cat $epilogueFile >> "$2/CMakeLists.txt"
#     else
#        cat $prologueFile > "$2/CMakeLists.txt"
#        echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc DST grapeFruit FLAGS --mlir-disable-threading --iree-quidditch-export-costs=$3 --iree-quidditch-import-tiles=$tile)" >> "$2/CMakeLists.txt"
#        echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc LLVM DST grapeFruit_llvm FLAGS --iree-quidditch-export-costs=$3 --iree-quidditch-import-tiles=$tile)" >> "$2/CMakeLists.txt"
#        cat $epilogueFile >> "$2/CMakeLists.txt"
#    fi
# }

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

echo -e "\tcompile.sh: generating the cmake files and compiling..."
echo -e "\tprocessing each point in the search space $basename"
uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
eatNum='^([0-9])([0-9])*'
for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
        do
        echo -e "\tGenerating source files for $ts..."
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
        echo -e "\t$M"
        echo -e "\t$N"
        echo -e "\t$K"
        echo -e "\t$m"
        echo -e "\t$n"
        echo -e "\t$k"
        dispatchNameTemplate="main\$async_dispatch_0_matmul_transpose_b_MxNxK_f64"
        dispatchName="${dispatchNameTemplate/MxNxK/"$M"x"$N"x"$K"}"
        echo -e "\tso the dispatch name is $dispatchName"
            # dims=$(echo $ts | grep -oE $dimsRegex) 
            # echo $dims
            # parts=(${(s/x/)$ts})
            # echo "$parts"
        # echo b=${ts:12:5}
            #a="56xhoodle"
            #echo ${ts#*x} 
            #echo "56xhoodle" | grep -oE $eatNum
            #echo "$ts" | grep -oE $dimsRegex
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
       
