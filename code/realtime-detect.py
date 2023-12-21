import mne
import numpy as np
import matplotlib.pyplot as plt
from Function.pre_process_data import *
from Function.Spike_manager_functions import *
from Function.parameter_setting_functions import *
from tqdm.auto import tqdm 
from pathlib import Path

# - Numpy
import torch
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous

# - Pretty printing
try:
    from rich import print
except:
    pass

# - Display images
from IPython.display import Image

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm

#获取模型
dilations = [2, 32]
n_out_neurons = 2
n_inp_neurons = 4
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 0.6
dt = 0.001
sim = WaveSenseNet(
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
sim.load('/home/ruixing/workspace/chbtar/chb/models/SNN_model_Isyn.pth')

#导入Samna
# - Import the Xylo HDK detection function
from rockpool.devices.xylo import find_xylo_hdks

# - Detect a connected HDK and import the required support package
connected_hdks, support_modules, chip_versions = find_xylo_hdks()

found_xylo = len(connected_hdks) > 0

if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'
spec = x.mapper(sim.as_graph(), weight_dtype = 'float')
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.config_from_specification(**spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = dt)
    print(modSamna)
    

#导入待测mne数据
sampling_frequency = 250
data = mne.io.read_raw_edf('raw_selected/raw_selected/train/chb01_03.edf',preload=True)
data = data.filter(l_freq=None,h_freq=30)
data = data.resample(sfreq=sampling_frequency)
tmin = 2985
tmax = 3045+5
data = data.crop(tmin=tmin,tmax=tmax)
picks1= ['C3-P3']
picks2= ['C4-P4']
one_sample_data = []
spike_array_one_trail = []
picks_list = list([picks1, picks2])
time_vector = np.linspace(0,5,1250)
interpfact = 250 #连续化时的采样率
refractory = 3e-4
dt = 0.01
sample = []
spike_array = []
epilepsy_test = []
io_power_list = []
logic_power_list = []
for j in picks_list:
    data_one_channel = data.get_data(units='uV',picks=j)
    one_sample_data.append(data_one_channel)
one_sample_data = np.asarray(one_sample_data)
array = one_sample_data.reshape(2,-1)
start =0
len_windows = 1250
step = 125
end = len_windows
array_encode = array.copy()
trigger = 0
for i in tqdm(range(array.shape[1]//125)):
    array_slices = array_encode[:,start:end]
    if array_slices.shape[1] != 1250:
        exit()
    array_slices.reshape(2,1250)
    for elc in np.arange(2):
        data_one_channel = array_slices[elc]
        ecog_signal, ecog_spikes = pre_process_data(raw_signal = data_one_channel,
                                                time_vector = time_vector, 
                                                interpfact = interpfact, 
                                                refractory = refractory)
        spike_list = {}
        spike_list['up'] = ecog_spikes['up']
        spike_list['dn'] = ecog_spikes['dn']
        spiketimes, neuronID = concatenate_spikes(spike_list)
        if spiketimes.shape[0] != 0:
            if spiketimes[-1] == (5.):
                spiketimes = np.delete(spiketimes,[-1])
                neuronID = np.delete(neuronID,[-1])
                
        if spiketimes.size == 0 or spiketimes.size == 1:
            synchronous_input = torch.zeros(1,500,2)
        else:
            # Get input signal 
            dt_original_data = 1/sampling_frequency
            num_timesteps = int(np.around(ecog_signal['time'][-1]+ dt_original_data - ecog_signal['time'][0], decimals = 3)* (1/dt))
            t_start  = ecog_signal['time'][0]
            t_stop = ecog_signal['time'][-1]
            asynchronous_input, synchronous_input = get_input_spikes(spiketimes = spiketimes,
                                                                neuronID = neuronID,
                                                                t_start = t_start, 
                                                                t_stop = t_stop,
                                                                num_timesteps = num_timesteps,
                                                                dt = dt)

        synchronous_input = synchronous_input.squeeze()
        synchronous_input = np.asarray(synchronous_input)
        synchronous_input = synchronous_input.T
        spike_array_one_trail.append(synchronous_input)
    spike_array_one_trail = np.asarray(spike_array_one_trail)
    spike_array_one_trail = spike_array_one_trail.reshape(4,500)
    start += step
    end += step
    #送进Samna
    tensor = torch.from_numpy(spike_array_one_trail.T)
    data = torch.tensor(tensor,dtype=torch.float)
    data = torch.reshape(data,(500,4))
    data = data.numpy()
    data = data.astype(int)
    modSamna.reset_state()
    # np.save('2_16_t1_test.npy',spike_array_one_trail)
    #************************************#
    out, _, recordings = modSamna((data*3).clip(0, 15),record=True,record_power = True,read_timeout = 20)
    #************************************#
    if np.size(recordings['io_power']) == 1 and np.size(recordings['logic_power']) == 1 :
        io_power_list.append(float(recordings['io_power']))
        logic_power_list.append(float(recordings['logic_power']))
    out = recordings['Isyn_out'].squeeze()
    peaks = out.max(0)
    result = peaks.argmax()
    print(f'现在进行到了{tmin+(start/sampling_frequency)}秒')
    epilepsy_test.append(result.item())
    print('peaks:',peaks)
    print(epilepsy_test)
    if result.item() == 1 and trigger == 0:
        print('检测到疑似癫痫')
    if result.item() == 1 and len(epilepsy_test)>=5 and all(x == 1 for x in epilepsy_test[-5:]):
        if trigger == 0:    
            print(f'在{tmin+(start/sampling_frequency-2.5)}秒检测到癫痫')
        trigger = 1 
        print('癫痫正在进行')
    if trigger == 1 and all(x == 0 for x in epilepsy_test[-5:]):
        trigger = 0
        print(f'癫痫在{tmin+(start/sampling_frequency-2.5)}时结束')
    spike_array_one_trail = []
print('完成')
# fig,ax = plt.subplot(2,figsize=(10,5))
# ax[0].plot(x1,y1)
# ax[1].plot(x1,y2)

