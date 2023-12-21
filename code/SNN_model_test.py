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
net.load('models/SNN_model_Isyn.pth')



net = net.to(device)

class Dataset(Dataset):
    def __init__(self,root_pos,root_neg):
        self.sample = []
        pos_dir = Path(root_pos)
        neg_dir = Path(root_neg)
        for i in sorted(pos_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_'):
                array = np.load(str(i),allow_pickle=True)
                tensor = torch.from_numpy(array.T)
                tensor = torch.tensor(tensor,dtype=torch.float)
                condititon = [tensor,torch.tensor(1)]
                self.sample.append(condititon)
                condititon = []
        for i in sorted(neg_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_'):
                array = np.load(str(i))
                tensor = torch.from_numpy(array.T)
                tensor = torch.tensor(tensor,dtype=torch.float)
                condititon = [tensor,torch.tensor(0)]*10
                self.sample.append(condititon)
                condititon = []
    
    def __getitem__(self,idx):
        data = self.sample[idx][0]
        label = self.sample[idx][1]
        return data,label
    
    def __len__(self):    # print('result:',result)
    # print('peaks:',peaks)
    # print('label:',label)
        return len(self.sample)
    
dataset_test = Dataset('data/test_data/spike/pos','data/test_data/spike/neg')
spiking_test_dataloader = DataLoader(dataset_test,batch_size=len(dataset_test),shuffle=True)
spiking_test_dataloader_unit = DataLoader(dataset_test,batch_size=1,shuffle=True)


#************#
#**整体预测***#
#************#
n_0 = 0
n_1 = 0
consequence_list = []
for data,label in tqdm(spiking_test_dataloader,colour='yellow'):
    data = data.to(device)
    label = label.to(device)
    # net.reset_state()
    data = torch.reshape(data,(len(dataset_test),500,4))
    # data = data.numpy()
    # data = data.astype(int)
    output, state, recordings = net(data,record=True)
    print(recordings.keys())
    # out = recordings['spk_out']['isyn'].squeeze()
    out = recordings['spk_out']['isyn'].squeeze()
    # print(out)
    peaks = out.max(1)[0]
    result = peaks.argmax(1)
    result = result.to(device)
    print('result:',result)
    print('label:',label)
    for i in result:
        if i.item() == 0:
            n_0  += 1

        if i.item() == 1:
            n_1  += 1
    consequence = (result==label)
    consequence_list.append(consequence)
    
acc = (consequence_list[0].sum())/consequence_list[0].shape[0]
print(f'准确率为:{acc}')
print(f'预测了{n_0}个0，{n_1}个1')


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
#     out = recordings['spk_out']['isyn'].squeeze()
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