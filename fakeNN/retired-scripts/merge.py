import sys
import pandas as pd
print("merge.py: ATTN: Run this script INSIDE directory Quidditch/comparing-tile-sizes/")

if len(sys.argv) != 4:
    print(f"USAGE: Requires 4 args.\nYou passed in {len(sys.argv)}")
else:
    print(f"merging {sys.argv[1]} {sys.argv[2]} ON {sys.argv[3]}")
    firstCSV=pd.read_csv(sys.argv[1])
    secondCSV=pd.read_csv(sys.argv[2])
    mergedCSV = pd.merge(firstCSV, secondCSV,on=sys.argv[3],how="inner")
    # print(mergedCSV)
    mergedCSV.to_csv("merged.csv",index=False)