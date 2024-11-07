# fits just right

`ConfigureForSnitch.cpp`:

```
// SmallVector<int64_t> l1Interchange = {2, 0, 1};
l1Tiles[0] = 0;
l1Tiles[1] = 40;
l1Tiles[2] = 100;
```

Right after `tensorTile.cpp` (L1 level):

```
 %26 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
 l1_tiles = [0, 40, 100], 
 l1_tiles_interchange = [2, 0, 1], 
 dual_buffer = true>} 
 ins(%extracted_slice, %extracted_slice_1 : tensor<1x100xf64>, tensor<40x100xf64>) 
 outs(%extracted_slice_2 : tensor<1x40xf64>) -> tensor<1x40xf64>
```

Right after `tensorTile.cpp` (Thread level):

```
 %34 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
 l1_tiles = [0, 40, 100], 
 l1_tiles_interchange = [2, 0, 1], dual_buffer = true>} 
 ins(%extracted_slice_8, %extracted_slice_9 : tensor<1x100xf64>, tensor<5x100xf64>) 
 outs(%extracted_slice_10 : tensor<1x5xf64>) -> tensor<1x5xf64>        
```

During `LowerL1Allocations.cpp`:

```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Let's look at ALL the allocOps before doing ANYTHING! 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 100 

allocOp with memref shape 40 100 

allocOp with memref shape 40 100 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1200 
Well, those were all the allocOps... =_=
```

