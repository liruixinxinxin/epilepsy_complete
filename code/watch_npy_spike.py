import numpy as np
from pathlib import Path
import torch

train_pos_dir = Path('data/train_data/spike/pos')
train_neg_dir = Path('data/train_data/spike/neg')
test_pos_dir = Path('data/test_data/spike/pos')
test_neg_dir = Path('data/test_data/spike/neg')
test = Path('data/train_data/spike/pos_test')
condition = []
for i in sorted(train_pos_dir.rglob('*.npy')):
    # if (str(i.parts[-1][9:16]) != 'trail1_') :
    array = np.load(str(i))
    print(f'{i.parts[-1][3:8]},{i.parts[-1][9:16]},spike:{array.sum()}')
    condition.append([i.parts[-1][3:8],i.parts[-1][9:16],'spike:',array.sum()])
pass