# - Numpy
import numpy as np
import torch
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300
import time
# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
import torch
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
# - Pretty printing
try:
    from rich import print
except:
    pass

# - Display images
from IPython.display import Image

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')
# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.devices import xylo as x

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
dilations = [2, 32]
n_out_neurons = 2
n_inp_neurons = 4
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 0.6
dt = 0.001
net = WaveSenseNet(
    dilations=dilations,
    n_classes=n_out_neurons,
    n_channels_in=n_inp_neurons,#in_channel
    n_channels_res=n_neurons,
    n_channels_skip=n_neurons,
    n_hidden=n_neurons,
    kernel_size=kernel_size,
    bias=Constant(0.0),
    smooth_output=True,
    tau_mem=Constant(tau_mem),
    base_tau_syn=base_tau_syn,
    tau_lp=tau_lp,
    threshold=Constant(threshold),
    neuron_model=LIFTorch,
    dt=dt,
)
net.load('models/SNN_model_Isyn.pth')
# net = toyKWSNet()
net = net.to(device)



g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.vA2.config_from_specification(**spec)
modSim = x.vA2.XyloSim.from_config(config)
out, _, rec = modSim.evolve(input_raster=np.zeros((10, 16)))

class Dataset(Dataset):
    def __init__(self,root_pos):
        self.sample = []
        pos_dir = Path(root_pos)
        for i in sorted(pos_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_'):
                array = np.load(str(i),allow_pickle=True)
                tensor = torch.from_numpy(array.T)
                tensor = torch.tensor(tensor,dtype=torch.float)
                condititon = [tensor,torch.tensor(1)]
                self.sample.append(condititon)
                condititon = []

            
    def __getitem__(self,idx):
        data = self.sample[idx][0]
        label = self.sample[idx][1]
        return data,label
    
    def __len__(self):
        return len(self.sample)
    
dataset_test = Dataset('data/train_data/spike/neg')
spiking_test_dataloader = DataLoader(dataset_test,batch_size=1,shuffle=True)

n = 0
right = 0
consequence_list = []
net1 = net

#pos
dir =Path('data/train_data/spike/pos')
for i in sorted(dir.rglob('*.npy')):
        modSim.reset_state()
        # print(net1 == net)
        data = np.load(str(i))
        data = torch.from_numpy(data.T)
        data = torch.tensor(data,dtype=torch.float)
        data = torch.reshape(data,(500,4))
        data = data.numpy()
        data = data.astype(int)
        # print(data[0:20])
        # if i.parts[-1] == 'chb02_16-trail1_pos_spike.npy':
        #     print('out:',out)
        # output, state, recordings = modSim((data*20).clip(0, 15),record=True,read_timeout=10)
        output, state, recordings = modSim((data*3).clip(0, 15),record=True,read_timeout=10)
        out = recordings['Isyn_out'].squeeze()
        # print(np.any(out))
        peaks = out.max(0)
        result = peaks.argmax()
        # if i.parts[-1] == 'chb02_16-trail1_pos_spike.npy':
        print('name:',str(i.parts[-1]),'peaks',peaks,'result:',result)
        if result == 1:
            right += 1
        n += 1

print(right/n)
    

#neg
# dir =Path('data/train_data/spike/neg')
# for i in sorted(dir.rglob('*.npy')):
#         modSim.reset_state()
#         # print(net1 == net)
#         data = np.load(str(i))
#         data = torch.from_numpy(data.T)
#         data = torch.tensor(data,dtype=torch.float)
#         data = torch.reshape(data,(500,4))
#         data = data.numpy()
#         data = data.astype(int)
#         # print(data[0:20])
#         # if i.parts[-1] == 'chb02_16-trail1_pos_spike.npy':
#         #     print('out:',out)
#         # output, state, recordings = modSim((data*20).clip(0, 15),record=True,read_timeout=10)
#         output, state, recordings = modSim((data*3).clip(0, 15),record=True,read_timeout=10)
#         out = recordings['Isyn_out'].squeeze()
#         # print(np.any(out))
#         peaks = out.max(0)
#         result = peaks.argmax()
#         # if i.parts[-1] == 'chb02_16-trail1_pos_spike.npy':
#         print('name:',str(i.parts[-1]),'peaks',peaks,'result:',result)
#         if result == 0:
#             right += 1
#         n += 1

# print(right/n)