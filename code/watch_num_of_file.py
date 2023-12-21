from pathlib import Path

root1 = 'data/train_data/spike/neg'
dir = Path(root1)
for i,j in enumerate(dir.rglob('*.npy')):
    pass
print(f'{root1}共有{i+1}个文件')

root2 = 'data/train_data/spike/pos'
dir = Path(root2)
for i,j in enumerate(dir.rglob('*.npy')):
    pass
print(f'{root2}共有{i+1}个文件')

root3 = 'data/test_data/spike/neg'
dir = Path(root3)
for i,j in enumerate(dir.rglob('*.npy')):
    pass
print(f'{root3}共有{i+1}个文件')

root4 = 'data/test_data/spike/pos'
dir = Path(root4)
for i,j in enumerate(dir.rglob('*.npy')):
    pass
print(f'{root4}共有{i+1}个文件')