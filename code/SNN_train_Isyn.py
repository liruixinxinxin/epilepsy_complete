import torch
import numpy as np
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Linear, Tanh
from rockpool.nn.modules import LIFTorch, LIFBitshiftTorch,ExpSynTorch
from rockpool.nn.modules import TorchModule, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool import TSContinuous
from rockpool.parameters import Constant
from pathlib import Path
from tqdm.auto import tqdm
from Function.parameter_setting_functions import *
from torch.optim import Adam, SGD
from rockpool.nn.modules import LIFTorch  # , LIFSlayer
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.nn.networks import SynNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                condititon = [tensor,torch.tensor(0)]
                self.sample.append(condititon)
                condititon = []
                
            
    def __getitem__(self,idx):
        data = self.sample[idx][0]
        label = self.sample[idx][1]
        return data,label
    
    def __len__(self):
        return len(self.sample)

batch_size = 128
dataset = Dataset('data/train_data/spike/pos',
                  'data/train_data/spike/neg')
dataset_test = Dataset('data/test_data/spike/pos','data/test_data/spike/neg')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
loader_one_epochs = DataLoader(dataset, batch_size=len(dataset), shuffle=False,drop_last=True)
spiking_test_dataloader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)

#=========================================================================================#
# Prepare SNN
#========================================================================================
dilations = [2, 32]
n_out_neurons = 2
n_inp_neurons = 4
n_neurons = 32
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


net = net.to(device)


#=========================================================================================#
# SNN is prepared 
#=========================================================================================#
# - Get the optimiser functions
optimizer = Adam(net.parameters().astorch(), lr=5e-4)
# optimizer = torch.optim.SGD(net.parameters().astorch(), lr=0.001, momentum=0.9)
# - Loss function
loss_fun = torch.nn.CrossEntropyLoss()
loss_fun = loss_fun.to(device)
# - Record the loss values over training iterations
loss_t = []
num_epochs = 300
Fig,ax = plt.subplots(3,figsize=(10,5))

train_accuracy = []
test_accuracy = []
# - Loop over iterations
loss_avg_list = []
def get_spike_count(rec, count=0):
    if 'spikes' in rec.keys():
        avg_spikes = torch.nn.functional.relu(rec['spikes'] - 10).mean()
        return count + avg_spikes

    for name, val in rec.items():
        if "output" in name:
            continue
        count = get_spike_count(val, count)

    return count
def reg_latency(Isyn, label, peaks, factor=2):
    res = torch.zeros_like(Isyn[:, 0])
    max_ind = torch.max(Isyn, dim=1)[1]
    for i in range(max_ind.size(0)):
        if peaks.argmax(1)[i] == label[i]:
            res[i, label[i]] = max_ind[i, label[i]]/Isyn.size(1)
        else:
            res[i, label[i]] = 1
    res = torch.sum(factor*res)
    return res
#计算tensor中最大数的索引

for n in tqdm(range(num_epochs)):
    loss_list = []
    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        net.reset_state()
        optimizer.zero_grad()
        data = torch.reshape(data,(batch_size,500,4))
        output, state, recordings = net(data,record=True)
        out = recordings['spk_out']['isyn'].squeeze()
        peaks = out.max(1)[0]
        peaks = peaks.to(device)
        pre_value = peaks.argmax(1)
        reg_lat = reg_latency(out, label, peaks)
        reg = (1 * (get_spike_count(recordings) ** 2))
        sub_loss = loss_fun(peaks, label) + reg_lat
        loss = sub_loss + reg
        loss.backward()
        print('pre_val:',pre_value.sum()/batch_size)
        result  = batch_size - ((label == pre_value).sum())
        print(F'预测错了{result}个,损失为{sub_loss}')
        optimizer.step()
        loss_list.append(sub_loss)
        pass
    pass
    with torch.no_grad():
        # net.eval()
        loss_avg = np.mean(np.asarray(loss_list))
        loss_avg_list.append(loss_avg)
        loss_list = []
        # print(f"loss_avg:{loss_avg}")
        print(f"the result of number of {n+1} is:")
        loader2 = DataLoader(dataset, batch_size=len(dataset),drop_last=True,shuffle=True)
        k = 1
        for data, label in loader2:
            data = data.to(device)
            label = label.to(device)
            net.reset_state()
            data = torch.reshape(data,(len(dataset),500,4))
            output, state, recordings = net(data,record=True)
            # out = recordings['spk_out']['isyn'].squeeze()
            out = recordings['spk_out']['isyn'].squeeze()
            peaks = out.max(1)[0]
            result = peaks.argmax(1)
            result = result.to(device)
            k = k+1
            n0 = 0
            n1 = 0
            for i in result:
                if i.item() == 0:
                    n0 += 1 
                if i.item() == 1 :
                    n1 += 1
            print(f'(training data)number of zero:{n0}，number of one:{n1}') 
            acc1 = (result==label).sum()/len(result)
            print(f'acc:{acc1}')
            train_accuracy.append(acc1)         
    #=========================================================================================#
    # test the SNN net
    #=========================================================================================#
    for data,label in tqdm(spiking_test_dataloader,colour='yellow'):
        data = data.to(device)
        label = label.to(device)
        net.reset_state()
        data = torch.reshape(data,(len(dataset_test),500,4))
        output, state, recordings = net(data,record=True)
        # out = recordings['spk_out']['isyn'].squeeze()
        out = recordings['spk_out']['isyn'].squeeze()
        peaks = out.max(1)[0]
        result = peaks.argmax(1)
        result = result.to(device)
        n0 = 0
        n1 = 0
        for i in result:
            if i.item() == 0:
                n0 += 1 
            if i.item() == 1 :
                n1 += 1
        acc2 = (result==label).sum()/len(result)
        print(f'(testing data)number of zero:{n0}，number of one:{n1}') 
        print(f'acc:{acc2}')
        test_accuracy.append(acc2)
    if (acc1>0.95 and acc2>0.96) or n == 299:
        # net.save('models/SNN_model_Isyn.pth')
        net.save('models/SNN_model_Isyn_synet.pth')
        print(acc1,acc2)
        x = np.linspace(1,n+1,n+1)
        ax[0].plot(x,train_accuracy)
        ax[0].set_title('training set accuracy curve')
        ax[1].plot(x,test_accuracy)
        ax[1].set_title('testing set accuracy curve')
        ax[2].plot(x,np.asarray(loss_avg_list))
        ax[2].set_title('average loss curve')
        plt.show()
        exit()