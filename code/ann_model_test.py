import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

ann = torch.load('models/ann_model.pth')
class Dataset(Dataset):
    def __init__(self,root_pos,root_neg):
        self.sample = []
        pos_dir = Path(root_pos)
        neg_dir = Path(root_neg)
        for i in sorted(pos_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_') and (str(i.parts[-1][9:16]) != 'trail2_'):
                array = np.load(str(i))
                tensor = torch.from_numpy(array.T)
                tensor = torch.tensor(tensor,dtype=torch.float)
                condititon = [tensor,torch.tensor(1)]
                self.sample.append(condititon)
                condititon = []
        for i in sorted(neg_dir.rglob('*.npy')):
            if (str(i.parts[-1][9:16]) != 'trail1_') and (str(i.parts[-1][9:16]) != 'trail2_'):
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
    
dataset = Dataset('data/test_data/spike/pos','data/test_data/spike/neg')
loader = DataLoader(dataset, batch_size=len(dataset),drop_last=True)

        

for data,label in iter(loader):
    data = torch.reshape(data,(len(dataset),1,500,4))
    output = ann(data)
    output = torch.reshape(output,(len(dataset),2))
    output = output.argmax(1)
    n0 = 0
    n1 = 0
    for i in output:
        if i == 0:
            n0 += 1 
        if i == 1 :
            n1 += 1
    print(output)
    spike = []
    for j in data:
        spike.append(j.sum())
    print(spike)
    print(f'预测了{n0}个neg，{n1}个pos') 
    print(f'准确率为:{(output==label).sum()/len(output)}')