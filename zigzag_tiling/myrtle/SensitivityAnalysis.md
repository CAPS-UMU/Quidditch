# Sensitivity Analysis

Tile Counts: https://docs.google.com/document/d/1fBikJNe1I9ZdXInhjIxX_nrJKJkuXH0A9UwOQW7gk-8/edit?usp=sharing

Cost Model: https://colab.research.google.com/drive/1fR_fq9LLSiY3LONu_zMSaZD2u1iee0EU?usp=sharing

## Results

| Tiling Scheme | NsNet kernel                 | Entire NN      | Inputs to Cost Model                                         | Estimated Cost |
| ------------- | ---------------------------- | -------------- | ------------------------------------------------------------ | -------------- |
| [0, 40, 100]  | 89788 cycles                 | 1112529 cycles | repeat = 5, fmadd = 1, fmadd_irr = 4, fld= 5, li = 0, iters=100, scf=1 |                |
| [0, 96, 40]   | doesn't fit in L1 :frowning: |                |                                                              |                |
| [0, 80, 40]   | 106194 cycles                | 1176454 cycles | repeat = 5, fmadd = 1, fmadd_irr = 4, fld = 5, li = 10, iters=40, scf=2 |                |
| [0, 40, 80]   | 107211 cycles                | 1181547 cycles | repeat = 5, fmadd = 1, fmadd_irr = 4, fld= 5, li = 0, iters=80, scf=1 |                |
| [0, 96, 32]   | 206234 cycles                | 1576329 cycles | repeat=4, fmadd=1, fmadd_irr=3, fld=4, li=8, iters=32, scf=3 |                |
| [0, 96, 20]   | 152769 cycles                | 1363936 cycles | repeat=4, fmadd=1, fmadd_irr=3, fld=4, li=8, iters=20, scf=3 |                |
| [0, 24, 120]  | 167284 cycles                | 1419432 cycles | repeat=3, fmadd=1, fmadd_irr=2, fld=3,li=0, iters=120, scf=1 |                |



## Tiling Schemes Analyzed

### [0, 40, 100]

```
snitch_stream.streaming_region {
    patterns = [
          #snitch_stream.stride_pattern<ub = [100], strides = [8], repeat = 5>,
          #snitch_stream.stride_pattern<ub = [100, 5], strides = [8, 800]>
    ]
}
```

```
      (%reinterpret_cast_1, %subview_16, %subview_17) 
      : memref<1x100xf64>, 
        memref<5x100xf64, strided<[100, 1], offset: ?>>, 
        memref<1x5xf64, strided<[1200, 1], offset: ?>>[{
        ...
        "    csrrsi zero, 1984, 1                         # SSR enable"
        "    mv t1, t0"
        "    fld ft7, 0(t1)                               # load double from memref of shape (1, 5)"
        "    fld ft6, 8(t0)                               # load double from memref of shape (1, 5)"
        "    fld ft5, 16(t0)                              # load double from memref of shape (1, 5)"
        "    fld ft4, 24(t0)                              # load double from memref of shape (1, 5)"
        "    fld ft3, 32(t0)                              # load double from memref of shape (1, 5)"
        "    li t1, 99"
        "    frep.o t1, 5, 0, 0"
        "    fmadd.d ft7, ft0, ft1, ft7"
        "    fmadd.d ft6, ft0, ft1, ft6"
        "    fmadd.d ft5, ft0, ft1, ft5"
        "    fmadd.d ft4, ft0, ft1, ft4"
        "    fmadd.d ft3, ft0, ft1, ft3"
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 1074619 - 1046815 = 27804
dispatch 0: 423335 - 405106 = 18229
dispatch 7: 970819 - 923733 = 47086
dispatch 8: 1043644 - 974019 = 69625
dispatch 1: 867063 - 777275 = 89788

cycles 1112529
```



### [0, 96, 40] (does not fit in L1)

```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Let's look at ALL the allocOps before doing ANYTHING! 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1248 

allocOp with memref shape 1 40 

allocOp with memref shape 96 40 

allocOp with memref shape 96 40 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1200 
Well, those were all the allocOps... =_=

allocOp with memref shape 1 1200 
memref size is 8
allocElements is 1200
NOW memref size is 9600
offset is 9600

allocOp with memref shape 1 1248 
memref size is 8
allocElements is 1248
NOW memref size is 9984
offset is 19584

allocOp with memref shape 1 40 
memref size is 8
allocElements is 40
NOW memref size is 320
offset is 19904

allocOp with memref shape 96 40 
memref size is 8
allocElements is 3840
NOW memref size is 30720
offset is 50624

allocOp with memref shape 96 40 
memref size is 8
allocElements is 3840
NOW memref size is 30720
offset is 81344

allocOp with memref shape 1 1200 
memref size is 8
allocElements is 1200
NOW memref size is 9600
offset is 90944

allocOp with memref shape 1 1200 
memref size is 8
allocElements is 1200
NOW memref size is 9600
offset is 100544

allocElements is 1200
memref size is 9600
offset is 100544
l1MemoryBytes is 100000, so 544 too much
kernel does not fit into L1 memory and cannot be compiled
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 11078473 - 11050661 = 27812
dispatch 0: 422041 - 403860 = 18181
dispatch 7: 10974411 - 10927492 = 46919
dispatch 8: 11047439 - 10977584 = 69855
dispatch 1: 0 - 0 = 0

cycles 11116309
```



### [0, 80, 40]

```
snitch_stream.streaming_region {
        patterns = [
          #snitch_stream.stride_pattern<ub = [2, 40], strides = [0, 8], repeat = 5>,
          #snitch_stream.stride_pattern<ub = [2, 40, 5], strides = [1600, 8, 320]>
        ]
} 
```

```
   (%reinterpret_cast_1, %subview_16, %subview_17) 
      : memref<1x40xf64>, 
        memref<10x40xf64, strided<[40, 1], offset: ?>>, 
        memref<1x10xf64, strided<[1200, 1], offset: ?>>[{
        ...
 "    li t2, 2" // for loop executes TWICE
        ...

"scf_body_0_for:"
"    li t4, 5"
"    mul t4, t1, t4"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
"    add t4, t0, t4"
"    fld ft7, 0(t4)                               # load double from memref of shape (1, 10)"
"    li t4, 5"
"    mul t4, t1, t4"
"    addi t4, t4, 1"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
"    add t4, t0, t4"
"    fld ft6, 0(t4)                               # load double from memref of shape (1, 10)"
"    li t4, 5"
"    mul t4, t1, t4"
"    addi t4, t4, 2"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
"    add t4, t0, t4"
"    fld ft5, 0(t4)                               # load double from memref of shape (1, 10)"
"    li t4, 5"
"    mul t4, t1, t4"
"    addi t4, t4, 3"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
"    add t4, t0, t4"
"    fld ft4, 0(t4)                               # load double from memref of shape (1, 10)"
"    li t4, 5"
"    mul t4, t1, t4"
"    addi t4, t4, 4"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
"    add t4, t0, t4"
"    fld ft3, 0(t4)                               # load double from memref of shape (1, 10)"

// hardware loop starts below
"    li t4, 39"
"    frep.o t4, 5, 0, 0"
"    fmadd.d ft7, ft0, ft1, ft7"
"    fmadd.d ft6, ft0, ft1, ft6"
"    fmadd.d ft5, ft0, ft1, ft5"
"    fmadd.d ft4, ft0, ft1, ft4"
"    fmadd.d ft3, ft0, ft1, ft3"
...
"    blt t1, t2, scf_body_0_for"
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 1138694 - 1110975 = 27719
dispatch 0: 421835 - 403618 = 18217
dispatch 7: 1034861 - 987813 = 47048
dispatch 8: 1107805 - 1038051 = 69754
dispatch 1: 931125 - 824931 = 106194

cycles 1176454
```



### [0, 40, 80]

```
snitch_stream.streaming_region {
    patterns = [
    #snitch_stream.stride_pattern<ub = [80], strides = [8], repeat = 5>,
    #snitch_stream.stride_pattern<ub = [80, 5], strides = [8, 640]>
    ]
} 
```

```
     (%reinterpret_cast_1, %subview_13, %subview_14) 
    : memref<1x80xf64>, 
      memref<5x80xf64, strided<[80, 1], offset: ?>>, 
      memref<1x5xf64, strided<[1200, 1], offset: ?>>[
      ...
      
     "    csrrsi zero, 1984, 1                         # SSR enable"
      "    mv t1, t0"
      "    fld ft7, 0(t1)                               # load double from memref of shape (1, 5)"
      "    fld ft6, 8(t0)                               # load double from memref of shape (1, 5)"
      "    fld ft5, 16(t0)                              # load double from memref of shape (1, 5)"
      "    fld ft4, 24(t0)                              # load double from memref of shape (1, 5)"
      "    fld ft3, 32(t0)                              # load double from memref of shape (1, 5)"
      "    li t1, 79"
      "    frep.o t1, 5, 0, 0"
      "    fmadd.d ft7, ft0, ft1, ft7"
      "    fmadd.d ft6, ft0, ft1, ft6"
      "    fmadd.d ft5, ft0, ft1, ft5"
      "    fmadd.d ft4, ft0, ft1, ft4"
      "    fmadd.d ft3, ft0, ft1, ft3"
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 1143443 - 1115742 = 27701
dispatch 0: 422859 - 404641 = 18218
dispatch 7: 1039990 - 992820 = 47170
dispatch 8: 1112509 - 1043253 = 69256
dispatch 1: 936170 - 828959 = 107211

cycles 1181547
```

### [0, 96, 20]

```
snitch_stream.streaming_region {
patterns = [
#snitch_stream.stride_pattern<ub = [3, 20], strides = [0, 8], repeat = 4>,
 #snitch_stream.stride_pattern<ub = [3, 20, 4], strides = [640, 8, 160]>
]
} 
```

```
...
"    li t2, 3"
"    mv t1, zero"
"    # Constant folded riscv_cf.bge"
"scf_body_0_for:"
"    li t4, 4"
"    mul t4, t1, t4"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
"    add t4, t0, t4"
"    fld ft6, 0(t4)                               # load double from memref of shape (1, 12)"
"    li t4, 4"
"    mul t4, t1, t4"
"    addi t4, t4, 1"
"    li t5, 8"
"    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft5, 0(t4)                               # load double from memref of shape (1, 12)"
        "    li t4, 4"
        "    mul t4, t1, t4"
        "    addi t4, t4, 2"
        "    li t5, 8"
        "    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft4, 0(t4)                               # load double from memref of shape (1, 12)"
        "    li t4, 4"
        "    mul t4, t1, t4"
        "    addi t4, t4, 3"
        "    li t5, 8"
        "    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft3, 0(t4)                               # load double from memref of shape (1, 12)"
"    li t4, 19"
"    frep.o t4, 4, 0, 0"
"    fmadd.d ft6, ft0, ft1, ft6"
"    fmadd.d ft5, ft0, ft1, ft5"
"    fmadd.d ft4, ft0, ft1, ft4"
"    fmadd.d ft3, ft0, ft1, ft3"
        ...
        "    blt t1, t2, scf_body_0_for"
        "scf_body_end_0_for:"
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 1325899 - 1298240 = 27659
dispatch 0: 422952 - 404724 = 18228
dispatch 7: 1222395 - 1175224 = 47171
dispatch 8: 1294994 - 1225720 = 69274
dispatch 1: 1118611 - 965842 = 152769

cycles 1363936
```



### [0, 24, 120]

```
snitch_stream.streaming_region {
patterns = [
#snitch_stream.stride_pattern<ub = [120], strides = [8], repeat = 3>,
#snitch_stream.stride_pattern<ub = [120, 3], strides = [8, 960]>
]
}
```

```
        "    csrrsi zero, 1984, 1                         # SSR enable"
        "    mv t1, t0"
        "    fld ft5, 0(t1)                               # load double from memref of shape (1, 3)"
        "    fld ft4, 8(t0)                               # load double from memref of shape (1, 3)"
        "    fld ft3, 16(t0)                              # load double from memref of shape (1, 3)"
        "    li t1, 119"
 // hardware loop starts here
       "    frep.o t1, 3, 0, 0"
        "    fmadd.d ft5, ft0, ft1, ft5"
        "    fmadd.d ft4, ft0, ft1, ft4"
        "    fmadd.d ft3, ft0, ft1, ft3"
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 1381529 - 1353677 = 27852
dispatch 0: 420140 - 401937 = 18203
dispatch 7: 1277331 - 1230209 = 47122
dispatch 8: 1350494 - 1280537 = 69957
dispatch 1: 1173559 - 1006275 = 167284

cycles 1419432
```



### [0, 96, 32]

```
snitch_stream.streaming_region {
patterns = [
#snitch_stream.stride_pattern<ub = [3, 32], strides = [0, 8], repeat = 4>,
#snitch_stream.stride_pattern<ub = [3, 32, 4], strides = [1024, 8, 256]>
]
} 
```

```
...
"    li t2, 3"
        "    mv t1, zero"
        "    # Constant folded riscv_cf.bge"
        "scf_body_0_for:"
        "    li t4, 4"
        "    mul t4, t1, t4"
        "    li t5, 8"
        "    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft6, 0(t4)                               # load double from memref of shape (1, 12)"
        "    li t4, 4"
        "    mul t4, t1, t4"
        "    addi t4, t4, 1"
        "    li t5, 8"
        "    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft5, 0(t4)                               # load double from memref of shape (1, 12)"
        "    li t4, 4"
        "    mul t4, t1, t4"
        "    addi t4, t4, 2"
        "    li t5, 8"
        "    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft4, 0(t4)                               # load double from memref of shape (1, 12)"
        "    li t4, 4"
        "    mul t4, t1, t4"
        "    addi t4, t4, 3"
        "    li t5, 8"
        "    mul t4, t4, t5                               # multiply by element size"
        "    add t4, t0, t4"
        "    fld ft3, 0(t4)                               # load double from memref of shape (1, 12)"
        "    li t4, 31"
        "    frep.o t4, 4, 0, 0"
        "    fmadd.d ft6, ft0, ft1, ft6"
        "    fmadd.d ft5, ft0, ft1, ft5"
        "    fmadd.d ft4, ft0, ft1, ft4"
        "    fmadd.d ft3, ft0, ft1, ft3"
        ...
 "    blt t1, t2, scf_body_0_for"
 "scf_body_end_0_for:"
```

```
[fesvr] Wrote 36 bytes of bootrom to 0x1000
dispatch 9: 1538576 - 1510873 = 27703
dispatch 0: 421769 - 403551 = 18218
dispatch 7: 1434733 - 1387827 = 46906
dispatch 8: 1507647 - 1437954 = 69693
dispatch 1: 1331289 - 1125055 = 206234

cycles 1576329
```

