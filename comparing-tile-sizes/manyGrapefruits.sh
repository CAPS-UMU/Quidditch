echo "manyGrapefruits.sh: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/"
here=$(pwd) # save current directory so we can return to it
# script-specific constants
tileSizes="/home/hoppip/Quidditch/comparing-tile-sizes/tile-sizes-to-test/*.json"
prologueFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-prologue.txt"
middleFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-middle-original.txt"
epilogueFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-epilogue.txt"
# build-specific constants
grapefruitDir="/home/hoppip/Quidditch/runtime/samples/grapeFruit"
buildDir="/home/hoppip/Quidditch/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="/home/hoppip/Quidditch/toolchain/bin"

# debugging
function_name(){
    echo "yohoho $1"
}

# generate cmakelists.txt file given 
# 1. the tile sizes json (do NOT use a relative path!)
# 2. directory in which to save the cmakelists.txt file (do NOT use a relative path!)
# 3. filepath to save the tile size costs (do NOT use a relative path!)
gen_cmakelists(){
    # echo ""
    # echo "tile sizes file is $1"
    # echo "directory to save in is $2"
    if [[ "$1" == "original" ]]; 
    then 
       cat $prologueFile > "$2/CMakeLists.txt"
       cat $middleFile >> "$2/CMakeLists.txt"
       cat $epilogueFile >> "$2/CMakeLists.txt"
    else
       cat $prologueFile > "$2/CMakeLists.txt"
       echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc DST grapeFruit FLAGS --mlir-disable-threading --iree-quidditch-export-costs=$3 --iree-quidditch-import-tiles=$1)" >> "$2/CMakeLists.txt"
       echo "quidditch_module(SRC \${CMAKE_CURRENT_BINARY_DIR}/grapeFruit.mlirbc LLVM DST grapeFruit_llvm FLAGS --iree-quidditch-export-costs=$3 --iree-quidditch-import-tiles=$1)" >> "$2/CMakeLists.txt"
       cat $epilogueFile >> "$2/CMakeLists.txt"
   fi
}


if [[ "$1" == "cmake" ]]; 
then 
       echo "manyGrapefruits.sh: generating the cmake files and compiling..."
       for ts in $tileSizes
        do
        basename=`basename $ts | sed 's/[.][^.]*$//'` # note that $ts is a full file path
        rm --f -R $basename # delete previous outputs
        mkdir -p $basename # create a local folder for this set of tile sizes
        echo "$basename.json" # inform user we are about to start processing $basename.json
        exportedCostFile="$here/$basename/tilingCosts.json" # using full path here
        gen_cmakelists $ts $grapefruitDir $exportedCostFile # generate basename-specific CMakeLists.txt
        gen_cmakelists $ts $basename $exportedCostFile # save a copy of it in our local folder
        cd $buildDir
        cmake .. -GNinja \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &> "$here/$basename/cmakeOutput.txt"
        ninja -j 20 &> "$here/$basename/buildOutput.txt"
        grep "kernel does not fit into L1 memory and cannot be compiled" "$here/$basename/buildOutput.txt"
        cd $here
        # copy generated executable to local folder
        cp $grapefruitExec "$basename/GrapeFruit" # copy SRC to DST  
        done
        gen_cmakelists "original" $grapefruitDir $exportedCostFile # generate basename-specific CMakeLists.txt
       
else
        if [[ "$1" == "actuallyOnlyOne" ]];
        then            
            if [[ "$3" == "noRun" ]];
            then
                basename=`basename $2 | sed 's/[.][^.]*$//'` # note that $2 ends with .json
                echo "Compiling with tile sizes $basename ..."
                rm --f -R $basename # delete previous outputs
                mkdir -p $basename # create a local folder for this set of tile sizes
                echo "$basename.json" # inform user we are about to start processing $basename.json
                exportedCostFile="$here/$basename/tilingCosts.json" # using full path here
                gen_cmakelists "$here/tile-sizes-to-test/$2" $grapefruitDir $exportedCostFile # generate basename-specific CMakeLists.txt
                gen_cmakelists "$here/tile-sizes-to-test/$2" $basename $exportedCostFile # save a copy of it in our local folder
                cd $buildDir
                cmake .. -GNinja \
                -DCMAKE_C_COMPILER=clang \
                -DCMAKE_CXX_COMPILER=clang++ \
                -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &> "$here/$basename/cmakeOutput.txt"
                ninja -j 20 &> "$here/$basename/buildOutput.txt"
                grep "kernel does not fit into L1 memory and cannot be compiled" "$here/$basename/buildOutput.txt"
                cd $here
                # copy generated executable to local folder
                cp $grapefruitExec "$basename/GrapeFruit" # copy SRC to DST
                gen_cmakelists "original" $grapefruitDir $exportedCostFile # generate basename-specific CMakeLists.txt 
                echo "Finished compiling."
            else
                basename=`basename $2 | sed 's/[.][^.]*$//'` # note that $2 ends with .json
                echo "Running $basename ..."
                myExecutable="$here/$basename/GrapeFruit"
                cd $verilator
                rm -f "$here/$basename/logs"
                ./snitch_cluster.vlt $myExecutable > "$here/$basename/run_output.txt"
                cp -r logs "$here/$basename/logs"
                cd $here
            fi
        else
        
       echo "manyGrapefruits: running each tiling option using verilator..."
        for ts in $tileSizes
        do
        basename=`basename $ts | sed 's/[.][^.]*$//'` # note that $ts is a full file path
        echo "$basename ..."
        myExecutable="$here/$basename/GrapeFruit"
        cd $verilator
        rm -f "$here/$basename/logs"
        ./snitch_cluster.vlt $myExecutable > "$here/$basename/run_output.txt"
        cp -r logs "$here/$basename/logs"
        cd $here
        done
        fi
fi

#./home/hoppip/Quidditch/toolchain/bin/snitch_cluster.vlt /home/hoppip/Quidditch/build/runtime/samples/grapeFruit/GrapeFruit

# for ts in $tileSizes
# do
#   basename=`basename $ts | sed 's/[.][^.]*$//'`
#   exportedCostTile="$grapefruitDir/$basename.json-exported.json"
#   mkdir -p $basename
#   function_name $basename
#   gen_cmakelists $ts $grapefruitDir
#   gen_cmakelists $ts $basename
#   gen_cmakelists "original" $grapefruitDir
#   # cd build
#   # cmake command
#   # ninja -j 20
#   # copy SRC to DST
#   cp $grapefruitExec "$basename/GrapeFruit"
#   cp $exportedCostTile "$basename/tilingCosts.json"

#   #pandoc -f markdown -t html "$mdFile" -o "contentAsHTML/$basename.md.html"   
# done
