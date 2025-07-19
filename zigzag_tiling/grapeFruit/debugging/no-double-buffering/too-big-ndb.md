# too big to fit

Original:

```
%23 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
l1_tiles = [0, 40, 100], 
l1_tiles_interchange = [2, 0, 1], 
dual_buffer = true>} 
ins(%18, %19 : tensor<1x400xf64>, tensor<1200x400xf64>) 
outs(%22 : tensor<1x1200xf64>) -> tensor<1x1200xf64>
```

Input has

- input matrix `1x400`
- weight matrix `1200x400`
- output matrix `1x1200`

ConfigureForSnitch.cpp:

```
dualBuffer = false;
l1Interchange = {0, 2, 1}; 
l1Tiles[0] = 0;
l1Tiles[1] = 240;
l1Tiles[2] = 80;
```

Right after `tensorTile.cpp` (L1 level):

```
%26 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
l1_tiles = [0, 240, 80], 
l1_tiles_interchange = [0, 2, 1]>} 
ins(%extracted_slice, %extracted_slice_1 : tensor<1x80xf64>, tensor<240x80xf64>) 
outs(%extracted_slice_2 : tensor<1x240xf64>) -> tensor<1x240xf64>             
```

Right after `tensorTile.cpp` (Thread level):

```
%34 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<
l1_tiles = [0, 240, 80], 
l1_tiles_interchange = [0, 2, 1]>} 
ins(%extracted_slice_14, %extracted_slice_15 : tensor<1x80xf64>, tensor<30x80xf64>) 
outs(%extracted_slice_16 : tensor<1x30xf64>) -> tensor<1x30xf64>       
```

During `LowerL1Allocations.cpp`:

```
allocOp with memref shape 1 1200     // never used

allocOp with memref shape 1 80       // never used

allocOp with memref shape 240 80     // never used

allocOp with memref shape 1 1200     // used to copy L3's 1x1200 argument to this L1 memref

allocOp with memref shape 1 1200     // used to copy L3's other 1x1200 argument to this L1 memref
Well, those were all the allocOps... =_=

allocOp with memref shape 1 1200 
offset is 0

allocOp with memref shape 1 80 
offset is 9600

allocOp with memref shape 240 80 
offset is 10240

allocElements is 19200
memref size is 153600
offset is 163840
l1MemoryBytes is 100000, so 63840 too much
kernel does not fit into L1 memory and cannot be compiled

```

We expect allocation at L1 for 

- `tensor<1x80xf64>`
- `tensor<240x80xf64>`
- `tensor<1x240xf64>`  

Questions:

- why does the first `1x1200` alloca not seem to be used? maybe it refers to an output operand, so don't have to copy from L3 to L1 initially?
- why does the `1x80` alloca not seem to be used? dma_transfer targets a memref.view of L1 instead?
- why does the `240x80` alloca not seem to be used? dma_transfer targets a memref.view of L1 instead?
- why is there NOT a `1x240` alloca?? We take a subview of a memref.view of L1 instead.
- Last two `1x1200` alloca ops are for the add operation, right?

**Where do the argument to the linalg.matmul come from, if NOT the alloca statements??**

Context:

```
"quidditch_snitch.memref.microkernel"(%44, %57, %58) ({
        ^bb0(%arg10: memref<1x80xf64, #quidditch_snitch.l1_encoding>, %arg11: memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>, %arg12: memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>):
          "linalg.matmul_transpose_b"(%arg10, %arg11, %arg12)
```

### I. Input matrix: `%44` (1x80 L1 tile)

```
%0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>

%42 = "arith.constant"() <{value = 9600 : index}> : () -> index

%43 = "memref.view"(%0, %42) : (memref<100000xi8>, index) -> memref<80xf64>

%44 = "memref.reinterpret_cast"(%43) <{
operandSegmentSizes = array<i32: 1, 0, 0, 0>, 
static_offsets = array<i64: 0>, 
static_sizes = array<i64: 1, 80>, 
static_strides = array<i64: 80, 1>}> : (memref<80xf64>) -> memref<1x80xf64>     
```

Reading: where is the copy from L3 to L1?

```
%47 = "dma.start_transfer"(%41, %46) : (memref<1x80xf64, strided<[400, 1], offset: ?>>, memref<1x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token
```

where %42 is a `1x80` subview into L3's `1x400` input matrix

```
%41 = "memref.subview"(%20, %arg7) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 80>, static_strides = array<i64: 1, 1>}> : (memref<1x400xf64, strided<[400, 1], offset: ?>>, index) -> memref<1x80xf64, strided<[400, 1], offset: ?>>
```



### II. Weight matrix: `%57` (240x80 L1 tile)

```
%0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>

%50 = "arith.constant"() <{value = 10240 : index}> : () -> index

%51 = "memref.view"(%0, %50) : (memref<100000xi8>, index) -> memref<19200xf64>

%52 = "memref.reinterpret_cast"(%51) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 80, 1>}> : (memref<19200xf64>) -> memref<240x80xf64>

%28 = "quidditch_snitch.compute_core_index"() : () -> index
%56 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 30)>}> : (index) -> index
"scf.for"(%56, %1, %1) ({
^bb0(%arg9: index):

%57 = "memref.subview"(%52, %arg9) <{
 operandSegmentSizes = array<i32: 1, 1, 0, 0>, 
 static_offsets = array<i64: -9223372036854775808, 0>, 
 static_sizes = array<i64: 30, 80>, static_strides = array<i64: 1, 1>}> 
 : (memref<240x80xf64>, index) -> memref<30x80xf64, strided<[80, 1], offset: ?>, #quidditch_snitch.l1_encoding>       
```

Reading: where is the copy from L3 to L1?

```
%55 = "dma.start_transfer"(%48, %54) : (memref<240x80xf64, strided<[400, 1], offset: ?>>, memref<240x80xf64, strided<[80, 1]>, #quidditch_snitch.l1_encoding>) -> !dma.token     
```

where %48 is a `240x80` subview into L3's `1200x400` weight matrix.

```
%48 = "memref.subview"(%21, %arg8, %arg7) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 240, 80>, static_strides = array<i64: 1, 1>}> : (memref<1200x400xf64, strided<[400, 1], offset: ?>>, index, index) -> memref<240x80xf64, strided<[400, 1], offset: ?>>      
```



### III. Output matrix: `%58`

```
%0 = "quidditch_snitch.l1_memory_view"() : () -> memref<100000xi8>

%24 = "arith.constant"() <{value = 0 : index}> : () -> index

%25 = "memref.view"(%0, %24) : (memref<100000xi8>, index) -> memref<1200xf64>

%26 = "memref.reinterpret_cast"(%25) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 1200>, static_strides = array<i64: 1200, 1>}> : (memref<1200xf64>) -> memref<1x1200xf64>

%1 = "arith.constant"() <{value = 240 : index}> : () -> index
%2 = "arith.constant"() <{value = 1200 : index}> : () -> index
%5 = "arith.constant"() <{value = 0 : index}> : () -> index

"scf.for"(%5, %2, %1) ({
^bb0(%arg8: index):

%49 = "memref.subview"(%26, %arg8) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 240>, static_strides = array<i64: 1, 1>}> : (memref<1x1200xf64>, index) -> memref<1x240xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>

%28 = "quidditch_snitch.compute_core_index"() : () -> index
%56 = "affine.apply"(%28) <{map = affine_map<()[s0] -> (s0 * 30)>}> : (index) -> index
"scf.for"(%56, %1, %1) ({
^bb0(%arg9: index):

%58 = "memref.subview"(%49, %arg9) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 30>, static_strides = array<i64: 1, 1>}> : (memref<1x240xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>, index) -> memref<1x30xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>        
```

Writing: where is the copy from L1 to L3?

```
%36 = "dma.start_transfer"(%33, %23) : (memref<1x1200xf64, #quidditch_snitch.l1_encoding>, memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> !dma.token
 
```

where 

- %23 is L3's 1x1200 representing the output of the whole function (not the linalg.matmul)??

```
%23 = "hal.interface.binding.subspan"(%19) {alignment = 64 : index, binding = 2 : index, descriptor_type = #hal.descriptor_type<storage_buffer>, operandSegmentSizes = array<i32: 1, 0>, set = 0 : index} : (index) -> memref<1x1200xf64, strided<[1200, 1], offset: ?>>
  "memref.assume_alignment"(%23) <{alignment = 1 : i32}> : (memref<1x1200xf64, strided<[1200, 1], offset: ?>>) -> ()
```

- %33 is the output of the addition stored in L1.