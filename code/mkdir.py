#you can creat the empty folder to save data


from pathlib import Path

Path('./pdata').mkdir(parents=True,exist_ok=True)

Path('pdata/test_data').mkdir(parents=True,exist_ok=True)
Path('pdata/train_data').mkdir(parents=True,exist_ok=True)


Path('pdata/test_data/npy').mkdir(parents=True,exist_ok=True)
Path('pdata/test_data/spike').mkdir(parents=True,exist_ok=True)
Path('pdata/test_data/npy/pos').mkdir(parents=True,exist_ok=True)
Path('pdata/test_data/npy/neg').mkdir(parents=True,exist_ok=True)
Path('pdata/test_data/spike/pos').mkdir(parents=True,exist_ok=True)
Path('pdata/test_data/spike/neg').mkdir(parents=True,exist_ok=True)


Path('pdata/train_data/npy').mkdir(parents=True,exist_ok=True)
Path('pdata/train_data/spike').mkdir(parents=True,exist_ok=True)
Path('pdata/train_data/npy/pos').mkdir(parents=True,exist_ok=True)
Path('pdata/train_data/npy/neg').mkdir(parents=True,exist_ok=True)
Path('pdata/train_data/spike/pos').mkdir(parents=True,exist_ok=True)
Path('pdata/train_data/spike/neg').mkdir(parents=True,exist_ok=True)