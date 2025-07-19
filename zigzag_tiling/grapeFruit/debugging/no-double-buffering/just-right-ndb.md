Original:

```
%23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
l1_tiles = [0, 40, 100], 
l1_tiles_interchange = [2, 0, 1], 
dual_buffer = true>} 
ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) 
outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
```

Changes made to configureForSnitch:

```
dualBuffer = false;
```

Right after `tensorTile.cpp` (L1 level):

```
hoodle
```

Right after `tensorTile.cpp` (Thread level):

```
hoodle
```

During `LowerL1Allocations.cpp`:

```
allocOp with memref shape 1 1200 // not used!

allocOp with memref shape 1 100  // not used!

allocOp with memref shape 40 100  // not used!

allocOp with memref shape 1 1200  // used

allocOp with memref shape 1 1200  // used
Well, those were all the allocOps... =_=

allocOp with memref shape 1 1200 
offset is 0

allocOp with memref shape 1 100 
offset is 9600

allocOp with memref shape 40 100 
offset is 10432

allocOp with memref shape 1 1200 
offset is 42432

allocOp with memref shape 1 1200 
offset is 52032
```

What about AFTER `LowerL1Allocations.cpp`?? What do these alloca instructions turn into?
