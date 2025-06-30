echo -e "\trun.sh: ATTN: NEVER run this script directly; Run script run_linear_layer.sh instead."
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
fakeNNExec=$9
verilator=${10}
# constants defined by script, according to what I have in fakeNN right now
origCosts="fakeNN-tile-sizes-costs" # (.txt)
origScheme="fakeNN-tile-sizes"      # (.json)
origHeader="fakeNN_util"            # (.h)
# constant defined by script, according to the templates folder contents
cmakeEpilog="CMakeLists-epilog"
cHeaderProlog="fakeNN_util-prolog"

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

uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
eatNum='^([0-9])([0-9])*'
# check build status of entry requested by CSV
batchSize=5
counter=0
        # build each entry requested by CSV
        for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
                do
                counter=$((counter+1))
               # echo "counter is $counter"
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
                # if run_output DNE, run the entry
                runOutputFile="$finalOutputDir/$ts/run_output.txt"
                myExec="$finalOutputDir/$ts/FakeNN"
                myOutputDir="$finalOutputDir/$ts"
                res=$(ls $runOutputFile 2>/dev/null)
                #echo "res is $res"
                # if [[ $runOutputFile != $res ]]; 
                # then 
                    echo -e "\trun.sh: getting run output for $ts"
                    #(cd $myOutputDir; $verilator $myExec > "$runOutputFile") &
                    cd $verilator
                    echo "myExec is $myExec"
                    echo "runOutputFile is $runOutputFile"
                    # ./snitch_cluster.vlt "$myExec" &> "$runOutputFile" &
                    # (sleep 2; echo "hoodle" &> "$runOutputFile" )&
                    (./snitch_cluster.vlt "$myExec" > "$runOutputFile")&
                    wait
                    # wait
                    cd $here
                    # if (( $counter % $batchSize == 0 )); then
                    # wait
                    # echo "starting new batch..."
                    # fi
                # else
                #     echo -e "\trun.sh: using cached run output for $ts"
                # fi
          
                # if (( $counter % $batchSize == 0 )); then
                #     wait
                #     echo "starting new batch..."
                # fi
        done
# wait

# batchSize=1
# counter=0
#         # build each entry requested by CSV
#         for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
#                 do
#                 counter=$((counter+1))
#                # echo "counter is $counter"
#                 eatNum='^([0-9])([0-9])*'
#                 M=$(echo $ts | grep -oE $eatNum)
#                 tail=${ts#*x}
#                 N=$(echo $tail | grep -oE $eatNum)
#                 tail=${tail#*x}
#                 K=$(echo $tail | grep -oE $eatNum)
#                 tail=${tail#*w}
#                 m=$(echo $tail | grep -oE $eatNum)
#                 tail=${tail#*-}
#                 n=$(echo $tail | grep -oE $eatNum)
#                 tail=${tail#*-}
#                 k=$(echo $tail | grep -oE $eatNum)
#                 golden=$(echo "$M""x""$N""x""$K""w0-0-0")
#                 # if golden run_output DNE, run the golden entry
#                 goldenOutputFile="$goldenOutputDir/$golden/run_output.txt"
#                 myExec="$goldenOutputDir/$golden/FakeNN"
#                 myOutputDir="$goldenOutputDir/$golden"
#                 res=$(ls $goldenOutputFile 2>/dev/null)
#                 if [[ $goldenOutputFile != $res ]]; 
#                 then 
#                     echo -e "\trun.sh: getting run output for $golden"
#                    cd $verilator
#                    ./snitch_cluster.vlt $myExec > "$goldenOutputFile" &
#                    cd $here
#                    # (. spark-verilator.sh $verilator $myExec $myOutputDir $goldenOutputFile) &
#                     #(cd $myOutputDir; $verilator $myExec > "$goldenOutputFile") &
                  
#                 else
#                     echo -e "\trun.sh: using cached golden run output for $golden"
#                 fi
#                 if (( $counter % $batchSize == 0 )); then
#                     wait
#                     echo "starting new batch..."
#                 fi
#         done
# wait
# echo "done waiting"

# # build each entry requested by CSV
# for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
#         do
#         eatNum='^([0-9])([0-9])*'
#         M=$(echo $ts | grep -oE $eatNum)
#         tail=${ts#*x}
#         N=$(echo $tail | grep -oE $eatNum)
#         tail=${tail#*x}
#         K=$(echo $tail | grep -oE $eatNum)
#         tail=${tail#*w}
#         m=$(echo $tail | grep -oE $eatNum)
#         tail=${tail#*-}
#         n=$(echo $tail | grep -oE $eatNum)
#         tail=${tail#*-}
#         k=$(echo $tail | grep -oE $eatNum)
#         golden=$(echo "$M""x""$N""x""$K""w0-0-0")
#         # if build does not exist, create files and build
#         buildOutputFile="$finalOutputDir/$ts/CMakeLists.txt"
#         res=$(ls $buildOutputFile 2>/dev/null)
#         if [[ $buildOutputFile != $res ]]; 
#         then 
#             echo -e "\trun.sh: creating build for $ts"
#             myBuildDir="$finalOutputDir/$ts"
#             rm -r -f $myBuildDir 2> /dev/null
#             mkdir $myBuildDir
#             gen_cmakelists_and_source $ts $jsonOutputDir $fakeNNDir $M $N $K $m $n $k
#             # save a copy of the build
#             cp "$fakeNNDir/CMakeLists.txt" "$fakeNNDir/$origHeader.h" "$fakeNNDir/$origScheme.json" "$fakeNNDir/$origCosts.txt" -t $myBuildDir
#             # actually build
#             cd $buildDir
#             cmake .. -GNinja \
#             -DCMAKE_C_COMPILER=clang \
#             -DCMAKE_CXX_COMPILER=clang++ \
#             -DCMAKE_C_COMPILER_LAUNCHER=ccache \
#             -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
#             -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &> "$fakeNNDir/cmakeOutput.txt"
#             ninja -j 20 &> "$fakeNNDir/buildOutput.txt"
#             grep "kernel does not fit into L1 memory and cannot be compiled" "$fakeNNDir/buildOutput.txt"
#             grep "Troublesome file path is" "$fakeNNDir/buildOutput.txt"
#             cd $here
#             # save a copy of the cmake and ninja build output
#             cp "$fakeNNDir/cmakeOutput.txt" "$fakeNNDir/buildOutput.txt" -t $myBuildDir
#             # save a copy of the generated executable
#             cp  "$fakeNNExec" -t $myBuildDir
#         else
#             echo -e "\trun.sh: using cached build for $ts"
#         fi
#         # if golden build does not exist, create golden files and build
#         goldenOutputFile="$goldenOutputDir/$golden/CMakeLists.txt"
#         res=$(ls $goldenOutputFile 2>/dev/null)
#         if [[ $goldenOutputFile != $res ]]; 
#         then 
#             echo -e "\trun.sh: creating golden build for $golden"
#             myBuildDir="$goldenOutputDir/$golden"
#             rm -r -f $myBuildDir 2> /dev/null
#             mkdir $myBuildDir
#             gen_cmakelists_and_source $golden $goldenJsonOutputDir $fakeNNDir $M $N $K 0 0 0
#             cp "$fakeNNDir/CMakeLists.txt" "$fakeNNDir/$origHeader.h" "$fakeNNDir/$origScheme.json" "$fakeNNDir/$origCosts.txt" -t $myBuildDir
#             # actually build
#             cd $buildDir
#             cmake .. -GNinja \
#             -DCMAKE_C_COMPILER=clang \
#             -DCMAKE_CXX_COMPILER=clang++ \
#             -DCMAKE_C_COMPILER_LAUNCHER=ccache \
#             -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
#             -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake &> "$fakeNNDir/cmakeOutput.txt"
#             ninja -j 20 &> "$fakeNNDir/buildOutput.txt"
#             grep "kernel does not fit into L1 memory and cannot be compiled" "$fakeNNDir/buildOutput.txt"
#             grep "Troublesome file path is" "$fakeNNDir/buildOutput.txt"
#             cd $here
#             # save a copy of the cmake and ninja build output
#             cp "$fakeNNDir/cmakeOutput.txt" "$fakeNNDir/buildOutput.txt" -t $myBuildDir
#             # save a copy of the generated executable
#             cp  "$fakeNNExec" -t $myBuildDir
#         else
#             echo -e "\trun.sh: using cached golden  build for $golden"
#         fi
# done
