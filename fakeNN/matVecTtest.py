
import itertools
filename="matVecTtest.csv"
print("yodelayheehoooooo")


# n must be divisible by 8 to avoid padding
# n must be a multiple of 1,2,[3,4,5,6,7],8 (the n' value)
# no restictions on k for now

# FLOP/c on the Y axis vs n on the X-axis 
# (where n varies from 40 to 50? [40, 41, 42 ... 50]) and N varies to be the nearest multiple of n.

# FLOP/c on the Y axis vs k on the X-axis 
# (where k varies from 100 to 110? [100,101,102,...100]) and K varies to be the nearest multiple of k

n_range = list(range(32,64,8))
k_range = list(range(80,110,5))
print(f'n_range is {n_range}')
print(f'k_range is {k_range}')
cart = list(itertools.product(n_range,k_range))

n_loops = list(range(1,10,1))
k_loops = list(range(1,10,2))

print(f'n_loops is {n_loops}')
print(f'k_loops is {k_loops}')

tiles = []
testcount = 0
for n,k in cart:
    print(f'L1 tile ({n},{k})')
    N_range =[]
    for a,b in list(itertools.product([n],n_loops)):
        N_range.append(a*b)
    print(f'N_range is {N_range}')
    K_range=[]
    for a,b in list(itertools.product([k],k_loops)):
        K_range.append(a*b)
    print(f'K_range is {K_range}')
    combos=list(itertools.product([(n,k)],N_range,K_range))
    tiles.append(combos)
    print(combos)
    testcount = testcount + len(combos)
    print('')

print(f'which is a total of {testcount} runs')
print(f'which is a about {testcount*15/60/16} hours if each test takes 15 minutes and they are 16 at a time.')
    #print(cart)

f = open(filename,"w")
f.write("JSON Name,M,N,K,m,n,k\n")
for tile in tiles:
    for nk,N,K in tile:
        f.write(f'1x{N}x{K}w{1}-{nk[0]}-{nk[1]},1,{N},{K},0,{nk[0]},{nk[1]}')
        f.write("\n")


f.close()

# print(list(itertools.product(n_range,n_loops)))
# N_range =[]
# for a,b in list(itertools.product(n_range,n_loops)):
#     N_range.append(a*b)
# print(f'N_range is {N_range}')

# K_range=[]
# for a,b in list(itertools.product(k_range,k_loops)):
#     K_range.append(a*b)
# print(f'K_range is {K_range}')