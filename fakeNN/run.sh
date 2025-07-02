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

uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
eatNum='^([0-9])([0-9])*'

batchSize=16
counter=0
        # run each entry requested by CSV
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
                # if run_output DNE, run the entry
                runOutputFile="$finalOutputDir/$ts/run_output.txt"
                myExec="$finalOutputDir/$ts/FakeNN"
                myOutputDir="$finalOutputDir/$ts"
                res=$(ls $runOutputFile 2>/dev/null)
                if [[ $runOutputFile != $res ]]; 
                then
                    counter=$((counter+1))
                    echo -e "\trun.sh: getting run output for $ts"
                    cd $myOutputDir
                    nohup "$verilator/snitch_cluster.vlt" "$myExec" &> "$runOutputFile" &
                    cd $here
                else
                    echo -e "\trun.sh: getting run output for $ts (using cached output)"
                fi
                # if golden run_output DNE, run the golden entry
                # goldenOutputFile="$goldenOutputDir/$golden/run_output.txt"
                # File="$goldenOutputDir/$golden/run_output.txt"
                # myExec="$goldenOutputDir/$golden/FakeNN"
                # myOutputDir="$goldenOutputDir/$golden"
                # res=$(ls $goldenOutputFile 2>/dev/null)
                # if [[ $goldenOutputFile != $res ]]; 
                # then
                #     counter=$((counter+1))
                #     echo -e "\trun.sh: getting run output for $golden (golden)"
                #     cd $myOutputDir
                #     nohup "$verilator/snitch_cluster.vlt" "$myExec" &> "$goldenOutputFile" &
                #     cd $here                  
                # else
                #     echo -e "\trun.sh: getting run output for $golden (golden) (using cached output)"
                # fi
                if (( $counter % $batchSize == 0 )); then
                    wait &> /dev/null
                    echo -e "\trun.sh: starting new batch..."
                fi
        done
wait &> /dev/null
