import mne
import numpy as np
import matplotlib.pyplot as plt


from mne.preprocessing import (ICA)
data = mne.io.read_raw_edf('/home/ruixing/workspace/chbtar/chb/raw_selected/raw_selected/train/chb01_03.edf')


ica = ICA(n_components=20)
ica.fit(data)
ica.plot_components()

# ica.plot_properties(good_ch,picks=[0,1])
# ica.plot_properties(good_ch,picks=np.array(range(0,50)))
# ica