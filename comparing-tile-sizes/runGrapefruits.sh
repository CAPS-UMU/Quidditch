echo "runGrapefruits.sh: ATTN: This should only be run by run_experiment.sh"
here=$(pwd) # save current directory so we can return to it
# script-specific constants"
quidditchDir="/home/emily/Quidditch"
tileSizesToTest="$quidditchDir/comparing-tile-sizes/tile-sizes-to-test"
tileSizes="$quidditchDir/comparing-tile-sizes/tile-sizes-to-test/*.json"
prologueFile="$quidditchDir/comparing-tile-sizes/cmakelist-prologue.txt"
middleFile="$quidditchDir/comparing-tile-sizes/cmakelist-middle-original.txt"
epilogueFile="$quidditchDir/comparing-tile-sizes/cmakelist-epilogue.txt"
searchSpaceCSV="$here/$1"
compileOutputDirectory="$here/$2"
sample_run_output="$quidditchDir/comparing-tile-sizes/sample_run_output.txt"
# build-specific constants
grapefruitDir="$quidditchDir/runtime/samples/grapeFruit"
buildDir="$quidditchDir/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="$quidditchDir/toolchain/bin"


## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi


echo "runGrapefruits.sh: running executables with verilator..."
batchSize=10
counter=0
       for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
        do
        counter=$((counter+1))
        basename=$ts # TODO: rename basname as ts everywhere
        myExecutable="$compileOutputDirectory/$basename/GrapeFruit"
        echo $myExecutable 
        cd $verilator
        rm -R "$compileOutputDirectory/$basename/logs/"
        nohup ./snitch_cluster.vlt $myExecutable &> "$compileOutputDirectory/$basename/run_output.txt" &
        cp -r logs "$compileOutputDirectory/$basename/logs"
        cd $here
        if (( $counter % $batchSize == 0 )); then
        wait
        echo "starting new batch..."
        fi
        done
wait &> /dev/null     
