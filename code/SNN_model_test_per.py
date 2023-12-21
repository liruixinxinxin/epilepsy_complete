import numpy as np
import torch
from rockpool.transform import quantize_methods as q
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.nn.modules import LIFTorch, LIFBitshiftTorch,ExpSynTorch
from rockpool.nn.modules import TorchModule, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool import TSContinuous
from rockpool.parameters import Constant
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dilations = [2, 32]
n_out_neurons = 2
n_inp_neurons = 4
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 2.0
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
net.load('models/SNN_model_*10.pth')


# net = toyKWSNet()
# net.load('models/SNN_model_Toy.pth')
net = net.to(device)


    

array = np.load('data/train_data/spike/pos_test/chb02_16-trail1_pos_spike.npy')
tensor = torch.from_numpy(array.T)
data = torch.tensor(tensor,dtype=torch.float)
data = torch.reshape(data,(1,500,4))
output, state, recordings = net((data*20).clip(0, 15),record=True)
print(recordings.keys())
# out = recordings['spk_out']['isyn'].squeeze()
out = recordings['spk_out']['isyn'].squeeze()
# print(out)
peaks,_ = torch.max(out, dim=0)
peaks = peaks.unsqueeze(0)
result = peaks.argmax(1)
print('result:',result)



#************#
#**单个预测***#
#************#

# n_0 = 0
# n_1 = 0
# consequence_list = []
# out_list = []
# for data,label in tqdm(spiking_test_dataloader_unit,colour='yellow'):
#     data = data.to(device)
#     label = label.to(device)
#     net.reset_state()
#     data = torch.reshape(data,(1,500,4))
#     # data = data.numpy()
#     # data = data.astype(int)
#     output, state, recordings = net(data,record=True)
#     print(recordings.keys())
#     # out = recordings['spk_out']['isyn'].squeeze()
#     out = recordings['5_LIFTorch']['isyn'].squeeze()
#     out_list.append(out.sum().item())
#     peaks = out.max(0)[0]
#     result = peaks.argmax()
#     if out.sum().item() == 0.0:
#         print('peaks:',peaks)
#         print('result:',result)
#     print('result:',result)
#     print('label:',label)
#     if result.item() == 0:
#         n_0  += 1
#     if result.item() == 1:
#         n_1  += 1
#     consequence = (result.item()==label.item())
#     consequence_list.append(consequence)
# print(out_list)
# acc = (sum(consequence_list))/len(dataset_test)
# print(f'准确率为:{acc}')
# print(f'预测了{n_0}个0，{n_1}个1')
# print(net.parameters())