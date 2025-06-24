import argparse
import os
import random
import torch
import torch.nn as nn
from iree.turbine import aot
import numpy as np

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

WIN_LEN = 0.02
HOP_FRAC = 0.5
FS = 16000
MIN_GAIN = -80

n_fft = int(WIN_LEN * FS)
frame_shift = WIN_LEN * HOP_FRAC
n_hop = n_fft * HOP_FRAC
spec_size = n_fft // 2 + 1

parser = argparse.ArgumentParser(prog='iree-turbine')
parser.add_argument('output', nargs='?')
parser.add_argument('--frames', dest='frames', metavar='N', type=int, default=1, nargs='?')
parser.add_argument('--m', dest='mDim', metavar='Mdim', type=int, default=1, nargs='?')
parser.add_argument('--n', dest='nDim', metavar='Ndim', type=int, default=1, nargs='?')
parser.add_argument('--k', dest='kDim', metavar='Kdim', type=int, default=1, nargs='?')
parser.add_argument('--dtype', dest='dtype', metavar='F', choices=['f32', 'f64'], default='f32')
parser.add_argument('-dump', dest='dump', action='store_true', default=False)
args = parser.parse_args()

name_to_dtype = {
    'f32': torch.float32,
    'f64': torch.float64,
}
dtype = name_to_dtype[args.dtype]

# beware! the name of your nn.Module subclass matters!
# use camel case starting with a capital letter, and remember that given a name like FakeNN,
# cmake generates a static linked library for the module called
# compiled_fake_n_n_linked_llvm_cpu_library_query etc.
class FakeNN(nn.Module):
    def __init__(self, n_features, hidden_1):
        super().__init__()
        self.n_features = n_features
        self.hidden_1 = hidden_1
        # fc1
        self.fc1 = nn.Linear(n_features, hidden_1, dtype=dtype)
        # other
        self.eps = 1e-9

    def forward(self, stft_noisy):
        out = self._forward(stft_noisy)
        return out

    def _forward(self, stft_noisy):
        x = self.fc1(stft_noisy)
        return x

model = FakeNN(n_features=args.kDim, hidden_1=args.nDim)
model.train(False)


def with_frames(n_frames):
    size = 1, n_frames, model.n_features
    # beware! the name of your aot.CompiledModule subclass matters!
    # use camel case starting with a capital letter, and remember that given a name like FakeNN,
    # cmake generates a static linked library for the module called
    # compiled_fake_n_n_linked_llvm_cpu_library_query etc.
    print(f'it looks like the shape of input for withFrames is {aot.AbstractTensor(*size, dtype=dtype)}')
    class CompiledFakeNN(aot.CompiledModule):
        def main(self, x=aot.AbstractTensor(*size, dtype=dtype)):
            y= aot.jittable(model.forward)(
                x,
                constraints=[]
            )
            return y

    return CompiledFakeNN

# I'm not sure what Markus meant by n_frames, but it appears
# that n_frames is equivalent to the M dimension (row dimension) of the input.
exported = aot.export(with_frames(n_frames=args.mDim))
if args.dump:
    exported.print_readable()
else:
    exported.save_mlir(args.output)
    # everything after this point are Emily's notes!!! 
    # print(f'spec_size is {spec_size}')
    # print(f'M:{args.mDim} N:{args.nDim} K:{args.kDim}')
    # size = (1,1,model.n_features)
    # print(f'input is {aot.AbstractTensor(*size, dtype=dtype)}')
    # print(f"Model structure: {model}\n")
    # for nm in model.state_dict():
    #     print("\t",end='')
    #     print(f'{nm} has shape {model.state_dict()[nm].shape}')
    
    # I think n_frames is the M dimension!!!!
    # print("\nRunning FakeNN with example input...")
    # np_dtype = np.float32
    # if dtype == torch.float64:
    #     np_dtype = np.float64
    # exInput = np.full((args.mDim, args.kDim), 7,dtype=np_dtype)
    # exInput = np.array([[list([i+1 for i in range (0,args.kDim)])]],dtype=np_dtype)
    # exInput = torch.from_numpy(exInput)
    # print(f'shape of input: {exInput.shape}')
    # result= model(exInput)
    # print("YODEL")
    # print(result[0])
    # print(f'shape of output: {result[0].shape}')