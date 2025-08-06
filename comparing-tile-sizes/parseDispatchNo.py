import re
import sys
import pandas as pd

# arg 1 is dispatchName as a string
def main():
    # print("myrtle: ",end='')
    dispatchName = sys.argv[1]
    dispatchRegex=re.compile(r'main\$async_dispatch_(\d+)_matmul_transpose_b_(\d+)x(\d+)x(\d+)_f64')
    dispNo,M,N,K = dispatchRegex.search(dispatchName).groups()
    print( dispNo)

if __name__ == "__main__":
    main()