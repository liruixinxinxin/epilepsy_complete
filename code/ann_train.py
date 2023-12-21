import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from Function.detach_dataset import *

#set Dataset
class Dataset(Dataset):
    def __init__(self,root_pos,root_neg):
        self.sample = []
        pos_dir = Path(root_pos)
        neg_dir = Path(root_neg)
        for i in sorted(pos_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_'):
                array = np.load(str(i))
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

batch_size = 10
p = 0
q = 0
dataset = Dataset('data/train_data/spike/pos','data/train_data/spike/neg')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
dataset_test = Dataset('data/test_data/spike/pos','data/test_data/spike/neg')
loader_test = DataLoader(dataset_test,batch_size=1,shuffle=True)


ann = nn.Sequential(
    nn.Conv2d(1, 20, (5,2), 1, bias=False),
    nn.ReLU(),
    nn.Conv2d(20, 32, (5,1), 1, bias=False),
    nn.ReLU(),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(47232, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 2, bias=False),
)

n_epochs = 50
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(ann.parameters(), lr=1e-4)
acc_train = []
acc_test = []
plt.figure(0,figsize=(10,5))
for n in range(n_epochs):
    for data, label in tqdm(iter(loader),colour='yellow'):
        optim.zero_grad()
        data = torch.reshape(data,(batch_size,1,500,4))
        output = ann(data)
        output2 = torch.reshape(output,(batch_size,2))
        loss = loss_fn(output2,label)
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        print(f"the result of number of {n}:")
        loader2 = DataLoader(dataset, batch_size=len(dataset),drop_last=True)
        for data,label in iter(loader2):
            data = torch.reshape(data,(len(dataset),1,500,4))
            output = ann(data)
            # output = output.mean(1)
            output = torch.reshape(output,(len(dataset),2))
            output = output.argmax(1)
            n0 = 0
            n1 = 0
            for i in output:
                if i == 0:
                    n0 += 1 
                if i == 1 :
                    n1 += 1
            print(f'num_0:{n0}，num_1:{n1}') 
            print(f'acc of train_dataset:{(output==label).sum()/len(output)}')
            acc_train.append((output==label).sum()/len(output))
    with torch.no_grad():
        z = 0
        o = 0
        for data,label in iter(loader_test):
            if label.item() == 0:
                z += 1
            if label.item() == 1:
                o += 1
        print(f'{z}个0，{o}个1')
        n_0 = 0
        n_1 = 0
        result_list = []
        num_result = 0
        for data,label in tqdm(loader_test,colour='yellow'):
            data = torch.reshape(data,(1,1,500,4))
            output = ann(data)
            # output = output.mean(1)
            output = output.reshape(1,2)
            output = output.argmax(1)
            result = (output.item()==label.item())
            if output.item() == 0:
                n_0  += 1
            if output.item() == 1:
                n_1 += 1
            result_list.append(result)
            if result == False:
                num_result += 1
        print(f'num_of_wrong:{num_result}')
        acc = sum(result_list)/len(result_list)
        print(f'acc:{acc}')
        print(f'num_0:{n0}，num_1:{n1}') 
        acc_test.append(acc)
torch.save(ann,'/home/ruixing/workspace/chbtar/chb/models/ann_model.pth')
plt.plot(range(n_epochs),acc_train)
plt.plot(range(n_epochs),acc_test)
plt.show()