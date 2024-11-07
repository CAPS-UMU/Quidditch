# too big to fit

ConfigureForSnitch.cpp:

```
l1Interchange = {0, 2, 1}; 
l1Tiles[0] = 0;
l1Tiles[1] = 240;
l1Tiles[2] = 80;
```

Right after `tensorTile.cpp` (L1 level):

```
%26 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
l1_tiles = [0, 240, 80], 
l1_tiles_interchange = [0, 2, 1], 
dual_buffer = true>} 
ins(%extracted_slice, %extracted_slice_1 : tensor<1x80xf64>, tensor<240x80xf64>) 
outs(%extracted_slice_2 : tensor<1x240xf64>) -> tensor<1x240xf64>           
```

Right after `tensorTile.cpp` (Thread level):

```
%34 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
l1_tiles = [0, 240, 80], 
l1_tiles_interchange = [0, 2, 1], 
dual_buffer = true>} 
ins(%extracted_slice_8, %extracted_slice_9 : tensor<1x80xf64>, tensor<30x80xf64>) 
outs(%extracted_slice_10 : tensor<1x30xf64>) -> tensor<1x30xf64>       
```

During `LowerL1Allocations.cpp`:

```
<eval_with_key>.0 from /home/hoppip/Quidditch/venv/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:551 in wrapped:19:0: warning: Let's look at ALL the allocOps before doging ANYTHING! 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 80 

allocOp with memref shape 240 80 

allocOp with memref shape 240 80 

allocOp with memref shape 1 1200 

allocOp with memref shape 1 1200 
Well, those were all the allocOps... =_=

```

We expect allocation at L1 for 

- `tensor<1x80xf64>`
- `tensor<240x80xf64>`
- `tensor<1x240xf64>`  

Questions:

- why do allocate `1x1200` and never seem to use it? I don't know.
- *no question about `1x80` allocation* This is used as an L1 tile.
- why do we allocate `240x80` and never seem to use it? I don't know.
- why do we allocate `240x80` a second time? This is used as an L1 tile.
- why do we NEVER allocate a `1x240`? Somehow a subview of L1 is taken directly? Annotate the source file to figure it out! OR, rerun example WITHOUT double buffering??
- why do we allocate a `1x1200` a third time? Because of linalg.add!
- why do we allocate a `1x1200` a fourth time? Because of *double buffering??*
