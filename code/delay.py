# - Numpy
import numpy as np
import torch
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

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
sr = 250
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


g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.vA2.config_from_specification(**spec)
sim = x.vA2.XyloSim.from_config(config)

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
start = 0
NT = 10
end = 500
step = int(end/NT)

stop = 0
n = 0


n_0 = 0
n_1 = 0

time_list = []

#pos
# dataset = Dataset('data/test_data/spike/pos')
# for i in tqdm(range(len(dataset)),colour='cyan'):
#     slices = []
#     data = dataset[i][0]
#     while stop < 500:
#         stop += step
#         slice = data[start:stop, :]
#         slices.append(slice)
#     stop = 0                                                         
#     for j in range(len(slices)):
#         pred = [0] * NT
#         out, _, recordings = sim.evolve(((slices[j].numpy().astype(int))*2.5).clip(0, 15),record=True)
#         out = recordings['Isyn_out'].squeeze()
#         peaks = out.max(0)
#         result = peaks.argmax()
#         if result.item() == 1:    
#             pred[j] = 1
#             time = (5/NT)*(j+1)
#             time_list.append(time)
#             n += 1
#             break
#         if all(elem == 0 for elem in pred) and j == 9:
#             time_list.append('*')
# print(time_list,n,len(time_list))

# fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

# x = range(len(time_list))
# y = [6 if i == '*' else i for i in time_list]

# ax.plot(x, y, 'o', markersize=5, color='black', alpha=0.5, label='Correct Prediction')
# ax.plot(x, [6 if i == '*' else None for i in time_list], 'o', alpha=0.5,markersize=5, color='red', label='Wrong Prediction')

# ax.set_xlabel('Index', fontsize=14)
# ax.set_ylabel('Value', fontsize=14)

# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
# ax.spines['bottom'].set_linewidth(0.5)
# ax.spines['left'].set_linewidth(0.5)
# ax.tick_params(axis='both', direction='out', width=0.5, length=4, labelsize=12)

# ax.grid(linestyle='--', linewidth=0.5)

# ax.legend(frameon=False, fontsize=12,loc='center right')

# plt.show()
# fig.savefig('pos_delay',dpi = 2000)


dataset = Dataset('data/test_data/spike/neg')
for i in tqdm(range(len(dataset)),colour='cyan'):
    slices = []
    data = dataset[i][0]
    while stop < 500:
        stop += step
        slice = data[start:stop, :]
        slices.append(slice)
    stop = 0 
    for j in range(len(slices)):
        sim.reset_state()
        pred = [0] * NT
        out, _, recordings = sim.evolve(((slices[j].numpy().astype(int))*2.5).clip(0, 15),record=True)
        out = recordings['Isyn_out'].squeeze()
        peaks = out.max(0)
        result = peaks.argmax()
        if result.item() == 1:    
            pred[j] = 1
            time = (5/NT)*(j+1)
            time_list.append(time)
            n += 1
            break
        if all(elem == 0 for elem in pred) and j == 9:
            time_list.append(6)
print(time_list,n,len(time_list))

fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

x = range(len(time_list))
y = [ i for i in time_list]

ax.legend(frameon=False, fontsize=12)
ax.plot(x, [i if i != 6 else None for i in time_list], 'o', markersize=5, color='red', alpha=0.5, label='Wrong Prediction')
ax.plot(x, [6 if i == 6 else None for i in time_list], 'o', markersize=5, color='black', alpha=0.5,label='Correct Prediction')
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('Value', fontsize=14)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.tick_params(axis='both', direction='out', width=0.5, length=4, labelsize=12)

ax.grid(linestyle='--', linewidth=0.5)

ax.legend(frameon=False, fontsize=12,loc='center right')

plt.show()
fig.savefig('neg_delay',dpi = 2000)
