import sys
import json
import tile_size_generator as tsg
import re
import pickle
import pandas as pd
from sklearn.svm import SVC, SVR


# def rankBy(dfs, id, by, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_sorted["rank"] = range(1, int(df_sorted.shape[0] + 1))
#     return df_sorted

# def getBestX(dfs, id, by, x, lowIsGood):
#     df_sorted = dfs[id].sort_values(by=by, ascending=lowIsGood)
#     df_best_5 = df_sorted.iloc[range(0, x)]
#     return df_best_5

def tileSelection(csvFile):
    df = pd.read_csv(csvFile)
    file = open("/home/emily/Quidditch/myrtle/dispatch-8-svr.pickle", 'rb')
    svr=pickle.load(file)
    # cut in half based on L1 usage
    # ranked = rankBy(dfs, (dispNo, 1), "Kernel Time", True)
    df["Predicted Kernel Time"] = df.apply(lambda y: svr.predict([y[["Microkernel Count","Regular Loads","Reused Streaming Loads","Space Needed in L1","Row Dim","Reduction Dim"]]])[0], axis=1)
    ranked = df.sort_values("Predicted Kernel Time", ascending=True)
    print(ranked)
    print(ranked.iloc[0]["Row Dim"])
    m = 1
    n = int(ranked.iloc[0]["Row Dim"])
    k = int(ranked.iloc[0]["Reduction Dim"])
    dualBuffer = True
    return (m,n,k,dualBuffer)

   

def faker(dispatchName):
    m=-1 # only fake it for the unhandled edge case
    n=40
    k=100
    dualBuffer = True
    # nsnet bugs special cases
    if dispatchName == "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64":
        m = 0
        n = 40
        k = 0
        dualBuffer = False
    if dispatchName == "main$async_dispatch_9_matmul_transpose_b_1x161x600_f64":
            m = 0
            n = 56
            k = 100
    # fake NN special case (for now)
    if dispatchName == "main$async_dispatch_0_matmul_transpose_b_1x120x40_f64":
            m = 0
            n = 0
            k = 60
    return (m,n,k,dualBuffer)

def main():
    print("yodelayheehoooooo")
    f = open(sys.argv[1], 'r')
    dispatchName = f.read()
    f.close()
    print (dispatchName)
    dispatchRegex=re.compile(r'main\$async_dispatch_\d+_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64')
    M,N,K = dispatchRegex.search(dispatchName).groups()
   # query fake myrtle!
    m,n,k,dualBuffer = faker(dispatchName)
    if m == -1: # if not edge case, use real myrtle
        # generate options
        jen = tsg.TileSizeGenerator(int(N),int(K),dispatchName)
        options = jen.validOptions()
        sizeAndLoadInfo = jen.exportOptionsToCSV(f'{M}x{N}x{K}wm-n-k', 1, options)
        print(sizeAndLoadInfo)        
        m,n,k,dualBuffer = tileSelection(f'{M}x{N}x{K}wm-n-k_case1_searchSpace.csv')
   
    # default values
    with open(sys.argv[2], 'r') as file:
        data = json.load(file)
    node = {}    
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    
    # set node values and export to JSON
    node["tile-sizes"] = [[m], [n], [k]]
    node["dual-buffer"] = dualBuffer
    data[dispatchName]=node    
    f = open(sys.argv[2], "w")   # 'r' for reading and 'w' for writing
    f.write(f"{json.dumps(data)}")
    f.close()

   

if __name__ == "__main__":
    main()

