import mne
import numpy as np
import matplotlib.pyplot as plt


raw = mne.io.read_raw_edf('raw_selected/raw_selected/train/chb01_03.edf',preload=True)
tmin = 0
tmax = 2000
raw = raw.crop(tmin=tmin,tmax=tmax)
raw = raw.filter(l_freq=None,h_freq=40)
raw = raw.resample(250)
raw.plot(scalings=128e-6,duration=120)
plt.show()
data= raw.get_data()
print(data)
