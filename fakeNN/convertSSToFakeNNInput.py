import sys
import pandas as pd
import re
import os.path
print("\nconvertSSToFakeNNInput.py: ATTN: Run this script INSIDE directory Quidditch/fakeNN/")

# extract dispatch number and dimensions
    # dispatchRegex = re.compile(
    #     r"main\$async_dispatch_(\d+)_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64"
    # )
    # d, M, N, K = dispatchRegex.search(df["Kernel Name_y"][0]).groups()

if len(sys.argv) != 2:
    print("\t",end='')
    print(f"USAGE: Requires a search space csv file name")
else:
    searchSpaceDF=pd.read_csv(sys.argv[1])
    myRegex=re.compile(r"(\d+)x(\d+)x(\d+)wm-n-k_searchSpace.csv")
    M, N, K = myRegex.search(sys.argv[1]).groups()

    f = open(f'{M}x{N}x{K}wm-n-k-fakeNN.csv',"w")
    f.write("FakeNN JSON Name,M,N,K,m,n,k,JSON Name\n")
    for i in range(0, searchSpaceDF.shape[0]):
        m = searchSpaceDF["m Dim"][i]
        n = searchSpaceDF["Row Dim"][i]
        k=searchSpaceDF["Reduction Dim"][i]
        f.write(f'{M}x{N}x{K}w{m}-{n}-{k},{M},{N},{K},{m},{n},{k},{m}-{n}-{k}')
        f.write("\n")
    f.close()

    convert = pd.read_csv(f'{M}x{N}x{K}wm-n-k-fakeNN.csv')
    merged = pd.merge(convert,searchSpaceDF,on="JSON Name",how="inner")
    merged.to_csv(
            f'{M}x{N}x{K}wm-n-k-fakeNN.csv',
            index=False,
        )
       
       