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
        # self.hidden_2 = hidden_2
        # self.hidden_3 = hidden_3
        # fc1
        self.fc1 = nn.Linear(n_features, hidden_1, dtype=dtype)
        # # rnn
        # self.rnn1 = nn.GRU(input_size=hidden_1, hidden_size=hidden_2, num_layers=1, batch_first=True, dtype=dtype)
        # self.rnn2 = nn.GRU(input_size=hidden_2, hidden_size=hidden_2, num_layers=1, batch_first=True, dtype=dtype)
        # # fc2
        # self.fc2 = nn.Linear(hidden_2, hidden_3, dtype=dtype)
        # # fc3
        # self.fc3 = nn.Linear(hidden_3, hidden_3, dtype=dtype)
        # # fc4
        # self.fc4 = nn.Linear(hidden_3, n_features, dtype=dtype)
        # other
        self.eps = 1e-9

    def forward(self, stft_noisy, *state_in):
        mask_pred, *state_out = self._forward(stft_noisy, *state_in)
        return mask_pred, *state_out

    def _forward(self, stft_noisy, *state_in):
      #  print("we are inside the foward function")
        x = self.fc1(stft_noisy)
        # for some reason, the "compiled_fake_n_n_linked_llvm_cpu_library_query" inside FakeNNLLVM.h
        # does not get generated unless I pass these two intermediate tensor states inside the forward function
        # and modify them
        # TODO: Since I'm not using the global intermediate states, find a way to get rid of them but still
        # generate the "compiled_fake_n_n_linked_llvm_cpu_library_query" needed.
        state_out = [*state_in]
        sevens = torch.full(state_out[0].shape, 7,dtype=state_out[0].dtype)#from_numpy(np.full(state_out[0].shape, 7,state_out[0].dtype))
        state_out[0] = torch.add(state_out[0],sevens)
        #result_tuple= model(torch.from_numpy(blah),state1,state2)
        #y, state_out[0] = self.rnn1(x, state_in[0])
        # x, state_out[1] = self.rnn2(x, state_in[1])
        # x = self.fc2(x)
        
        # x = nn.functional.relu(x)
        # x = self.fc3(x)
        # x = nn.functional.relu(x)
        # x = self.fc4(x)
        # x = torch.sigmoid(x)
        # sort shape
        #mask_pred = x.permute(0, 2, 1).unsqueeze(1)
      #  print(f'inside the forward funcion, about to return{state_out}')
        return x, *state_out


model = FakeNN(n_features=args.kDim, hidden_1=args.nDim)
#print(model)
model.train(False)


def with_frames(n_frames):
    size = 1, n_frames, model.n_features
    # beware! the name of your aot.CompiledModule subclass matters!
    # use camel case starting with a capital letter, and remember that given a name like FakeNN,
    # cmake generates a static linked library for the module called
    # compiled_fake_n_n_linked_llvm_cpu_library_query etc.
    class CompiledFakeNN(aot.CompiledModule):
        # Make the hidden state globals that persist as long as the IREE session does.
        state1 = aot.export_global(torch.zeros(1, 1, 2, dtype=dtype), mutable=True, uninitialized=False)
        state2 = aot.export_global(torch.zeros(1, 1, 2, dtype=dtype), mutable=True, uninitialized=False)
        def main(self, x=aot.AbstractTensor(*size, dtype=dtype)):
            y, out1, out2 = aot.jittable(model.forward)(
                x, self.state1, self.state2,
                constraints=[]
            )
            self.state1 = out1
            self.state2 = out2
            return y

    return CompiledFakeNN

# I'm not sure what Markus meant by n_frames, but it appears
# that n_frames is equivalent to the M dimension (row dimension) of the input.
exported = aot.export(with_frames(n_frames=args.mDim))
if args.dump:
    exported.print_readable()
else:
    exported.save_mlir(args.output)
    # emily's notes below vvvvvvvvvvvvvvvvv
    # np_dtype = np.float32
    # if dtype == torch.float64:
    #     np_dtype = np.float64

    # blah = np.full((1,args.mDim, args.kDim), 7,dtype=np_dtype)
    # state1 = torch.zeros(1, 1, 400, dtype=dtype)
    # state2 = torch.zeros(1, 1, 400, dtype=dtype)
    # result_tuple= model(torch.from_numpy(blah),state1,state2)
    # print("YODEL")
    # print(result_tuple[0])
    # print(result_tuple[0].shape)


# THE FOLLOWING CODE RUNS THE NsNet2 NN IN PYTHON!!
    # np_dtype = np.float32
    # if dtype == torch.float64:
    #     np_dtype = np.float64
    #     print("yodelaheyyyhoooooooo")
    # blah = np.full((1,1, 161), 7,dtype=np_dtype)
    # blah = np.array([[list([i+1 for i in range (0,161)])]],dtype=np_dtype)
    # state1 = torch.zeros(1, 1, 400, dtype=dtype)
    # state2 = torch.zeros(1, 1, 400, dtype=dtype)
    # result_tuple= model(torch.from_numpy(blah),state1,state2)
    # print("YODEL")
    # print(result_tuple)
# https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
# https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://docs.kanaries.net/topics/Python/nn-linear