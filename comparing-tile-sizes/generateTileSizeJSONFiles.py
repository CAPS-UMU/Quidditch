import sys
import pandas as pd
import json
print("generateTileSizeJSONFiles.py: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/")

if len(sys.argv) != 4:
    print(f"USAGE: Requires a search space csv file, kernel name, and output folder path!.\nYou passed in {len(sys.argv)} args")
else:
    # set up default tiling scheme for each kernel
    data = {}
    # dispatch 1
    node = {}
    node["tile-sizes"] = [[0], [40], [100]]
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    node["dual-buffer"] = True
    data["main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64"]=node
    # dispatch 0
    node = {}
    node["tile-sizes"] = [[0], [40], [0]]
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    node["dual-buffer"] = False
    data["main$async_dispatch_0_matmul_transpose_b_1x400x161_f64"]=node
    # dispatch 7
    node = {}
    node["tile-sizes"] = [[0], [40], [100]]
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    node["dual-buffer"] = True
    data["main$async_dispatch_7_matmul_transpose_b_1x600x400_f64"]=node
    # dispatch 8
    node = {}
    node["tile-sizes"] = [[0], [40], [100]]
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    node["dual-buffer"] = True
    data["main$async_dispatch_8_matmul_transpose_b_1x600x600_f64"]=node
    # dispatch 9
    node = {}
    node["tile-sizes"] = [[0], [56], [100]]
    node["loop-order"] = [[2,0], [0,0], [1,0]]
    node["dual-buffer"] = True
    data["main$async_dispatch_9_matmul_transpose_b_1x161x600_f64"]=node
    # override tiling scheme for one kernel
    searchSpaceDF=pd.read_csv(sys.argv[1])
    for i in range(0, searchSpaceDF.shape[0]):
        theName = searchSpaceDF["JSON Name"][i]
        rowDim = searchSpaceDF["Row Dim"][i]
        redDim = searchSpaceDF["Reduction Dim"][i]
        jsonPath = f"{sys.argv[3]}/{theName}.json"
        print(f'writing to {jsonPath}')
        #jsonPath = f"{testerFolder}/{theName}.json"
        f = open(jsonPath, "w")   # 'r' for reading and 'w' for writing 
        data[f'{sys.argv[2]}']["tile-sizes"]=[[0], [int(rowDim)], [int(redDim)]]
        f.write(f"{json.dumps(data)}")
        f.close()  
   
    