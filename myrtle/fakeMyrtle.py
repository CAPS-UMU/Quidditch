import sys
import json

print("yodelayheehoooooo")
f = open(sys.argv[1], 'r')
file_contents = f.read()
print (file_contents)
f.close()

data = {}
node = {}
node["tile-sizes"] = [[0], [40], [100]]
node["loop-order"] = [[2,0], [0,0], [1,0]]
node["dual-buffer"] = True
#dispatchName = f'main$async_dispatch_0_matmul_transpose_b_{mC}x{nC}x{kC}_f64'
dispatchName=file_contents
data[dispatchName]=node
m=0
n=40
k=100

# nsnet bugs special cases
if dispatchName == "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64":
    m = 0
    n = 40
    k = 0
    node["dual-buffer"] = False
if dispatchName == "main$async_dispatch_9_matmul_transpose_b_1x161x600_f64":
        m = 0
        n = 56
        k = 100
# fake NN specialc cases (for now)
if dispatchName == "main$async_dispatch_0_matmul_transpose_b_1x120x40_f64":
        m = 0
        n = 0
        k = 60

f = open(sys.argv[1], "w")   # 'r' for reading and 'w' for writing 
data[f'{dispatchName}']["tile-sizes"]=[[int(m)], [int(n)], [int(k)]]
f.write(f"{json.dumps(data)}")
f.close()