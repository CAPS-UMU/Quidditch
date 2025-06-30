import sys
import pandas as pd
import json
import os.path
print("\n\tgenerateTileSizeJSONFiles.py: ATTN: Run this script INSIDE directory Quidditch/fakeNN/")

if len(sys.argv) != 4:
    print("\t",end='')
    print(f"USAGE: Requires a search space csv file output folder path, and golden output folder path!\nYou passed in {len(sys.argv)} args")
else:
    # for each input size and tiling scheme
    # save a json tiling scheme given m, n, k
    # save a json tiling scheme with m=0, n=0, k=0 (golden)
    searchSpaceDF=pd.read_csv(sys.argv[1])
    for i in range(0, searchSpaceDF.shape[0]):
        theName = searchSpaceDF["JSON Name"][i]
        m = searchSpaceDF["m"][i]
        n = searchSpaceDF["n"][i]
        k=searchSpaceDF["k"][i]
        mC = searchSpaceDF["M"][i]
        nC = searchSpaceDF["N"][i]
        kC=searchSpaceDF["K"][i]
        # create json representation of tiling scheme
        data = {}
        node = {}
        node["tile-sizes"] = [[0], [40], [100]]
        node["loop-order"] = [[2,0], [0,0], [1,0]]
        node["dual-buffer"] = True
        dispatchName = f'main$async_dispatch_0_matmul_transpose_b_{mC}x{nC}x{kC}_f64'
        data[dispatchName]=node
        # if ts dne, generate it
        jsonPath = f"{sys.argv[2]}/{theName}.json"
        if not os.path.exists(jsonPath):
            print("\t",end='')
            print(f'writing to {jsonPath}')
            f = open(jsonPath, "w")   # 'r' for reading and 'w' for writing 
            data[f'{dispatchName}']["tile-sizes"]=[[int(m)], [int(n)], [int(k)]]
            f.write(f"{json.dumps(data)}")
            f.close()
        else:
            print("\t",end='')
            print(f'using cached {jsonPath}')
        # if golden ts dne, generate it
        theName=f'{mC}x{nC}x{kC}w{0}-{0}-{0}'
        jsonPath = f"{sys.argv[3]}/{theName}.json"
        if not os.path.exists(jsonPath):
            print("\t",end='')
            print(f'writing to {jsonPath}')
            f = open(jsonPath, "w")   # 'r' for reading and 'w' for writing 
            data[f'{dispatchName}']["tile-sizes"]=[[int(0)], [int(0)], [int(0)]]
            f.write(f"{json.dumps(data)}")
            f.close()
        else:
            print("\t",end='')
            print(f'using cached {jsonPath}')
   
    