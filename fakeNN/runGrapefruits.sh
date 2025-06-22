echo "runGrapefruits.sh: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/"
here=$(pwd) # save current directory so we can return to it
# script-specific constants"
tileSizesToTest="/home/hoppip/Quidditch/comparing-tile-sizes/tile-sizes-to-test"
tileSizes="/home/hoppip/Quidditch/comparing-tile-sizes/tile-sizes-to-test/*.json"
prologueFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-prologue.txt"
middleFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-middle-original.txt"
epilogueFile="/home/hoppip/Quidditch/comparing-tile-sizes/cmakelist-epilogue.txt"
searchSpaceCSV="$here/$1"
compileOutputDirectory="$here/$2"
sample_run_output="/home/hoppip/Quidditch/comparing-tile-sizes/sample_run_output.txt"
# build-specific constants
grapefruitDir="/home/hoppip/Quidditch/runtime/samples/grapeFruit"
buildDir="/home/hoppip/Quidditch/build"
grapefruitExec="$buildDir/runtime/samples/grapeFruit/GrapeFruit"
verilator="/home/hoppip/Quidditch/toolchain/bin"


## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi


echo "runGrapefruits.sh: running executables with verilator..."
batchSize=5
counter=0
       for ts in $(grep -oE '^(0-([0-9]*)-([0-9]*))' $searchSpaceCSV)
        do
        #sh sleepy.sh &
        counter=$((counter+1))
        basename=$ts # TODO: rename basname as ts everywhere
        myExecutable="$compileOutputDirectory/$basename/GrapeFruit"
        echo $myExecutable 
        cd $verilator
        rm -R "$compileOutputDirectory/$basename/logs/"
        ./snitch_cluster.vlt $myExecutable > "$compileOutputDirectory/$basename/run_output.txt" &
        cp -r logs "$compileOutputDirectory/$basename/logs"
        cd $here
        if (( $counter % $batchSize == 0 )); then
        wait
        echo "starting new batch..."
        fi
        done
      
