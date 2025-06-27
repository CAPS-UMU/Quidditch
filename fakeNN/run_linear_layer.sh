echo "run_linear_layer.sh: ATTN: Run this script INSIDE directory Quidditch/fakeNN/"
here=$(pwd) # save current directory so we can return to it
basename=`basename $1 | sed 's/[.][^.]*$//'` # strip search space argument of its .csv extension
# script-specific constants
searchSpaceCSV="$here/linear-layer-search-space/$basename.csv"
experimentName="$2"
finalOutputDirectory="$here/linear-layer-search-space/out"
jsonOutputDirectory="$here/linear-layer-search-space/tiling-schemes"

## this script requires a search space csv file
res=$(ls $searchSpaceCSV 2>/dev/null)
if [[ $searchSpaceCSV != $res ]]; 
    then 
    echo "ERROR: search space file $searchSpaceCSV not found!"
    exit 1
fi

echo "processing each point in the search space $basename"

uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
dimsRegex='([0-9]*|[x]*|[w]*|[-]*)*^(([0-9]*),([0-9]*),([0-9]*)),([0-9]*),([0-9]*),([0-9]*),'
inputSizeRegex='(([0-9]*)x)|([0-9]*)w'
hoodle='^([0-9])x([0-9]*|[x]*|[w]*|[-]*)*'
eatNum='^([0-9])([0-9])*'
for ts in $(grep -oE $uniquePointRegex $searchSpaceCSV)
        do
        echo "Generating source files for $ts..."
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
        echo "$M"
        echo "$N"
        echo "$K"
        echo "$m"
        echo "$n"
        echo "$k"
        dispatchNameTemplate="main\$async_dispatch_0_matmul_transpose_b_MxNxK_f64"
        dispatchName="${dispatchNameTemplate/MxNxK/"$M"x"$N"x"$K"}"
        echo "so the dispatch name is $dispatchName"
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


#"main\$async_dispatch_0_matmul_transpose_b_2x120x40_f64"
# ## generate json files
# if [[ "$3" == "genJsons" ]];
#     then
#     echo "run_experiment.sh: generating json files from the search space..."
#     mkdir -p $jsonOutputDirectory
#     python generateTileSizeJSONFiles.py $1 $8 $jsonOutputDirectory
# fi

# ## compile
# if [[ "$4" == "compile" ]];
#     then
#     ## compile
#     sh compileGrapefruits.sh $1 $experimentName
#     ## check compilation results
#     else if [[ "$4" == "status" ]];
#              then
#              sh compileGrapefruits.sh $1 $experimentName status
#         fi
# fi

# ## run
# if [[ "$5" == "run" ]];
#     then
#     sh runGrapefruits.sh $1 $experimentName
# fi


# ## export 
# if [[ "$6" == "correctness" ]];
#     then
#     sh scrutinizeGrapefruits.sh $1 $experimentName $7 $8
# fi
# if [[ "$6" == "export" ]];
#     then
#     sh scrapeGrapefruits.sh $1 $experimentName $7 $8
# fi


