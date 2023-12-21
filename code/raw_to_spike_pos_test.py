import mne
import numpy as np
import matplotlib.pyplot as plt
from Function.pre_process_data import *
from Function.Spike_manager_functions import *
from Function.parameter_setting_functions import *
from tqdm.auto import tqdm 
from pathlib import Path

picks1= ['C3-P3']
picks2= ['C4-P4']
picks_list = list([picks1, picks2])


time_dict ={
 'chb20_12': [104, 123],'chb20_13': [2498, 2537], 'chb20_16': [2226, 2261], 'chb20_68': [1405, 1432],  
 
 'chb21_19': [1288,1344],'chb21_20': [2627, 2677], 'chb21_21': [2003, 2084],
 'chb21_22': [2553, 2565],
  
 'chb22_20': [3367, 3425], 'chb22_25': [3139, 3213], 'chb22_38': [1263, 1335], 
  
 'chb23_06': [3975, 4075], 'chb23_08': [5104, 5151],'chb23_09': [2610, 2660], 
 
 'chb24_01': [2451, 2470], 'chb24_03': [2883, 2908], 'chb24_04': [1088, 1122], 'chb24_06': [1229, 1245], 
 'chb24_07': [38, 60],'chb24_09': [1745, 1760],'chb24_11': [3567, 3597],'chb24_13': [3288, 3304],'chb24_14': [1939, 1955],
 'chb24_15': [3552, 3569],'chb24_17': [3515, 3581],'chb24_21': [2804, 2872],
 }

dir = Path('raw_selected/raw_selected/test')
for i in tqdm(sorted(dir.rglob('*.edf'))):
    data = mne.io.read_raw_edf(str(i),preload=True)
    tmin = int((time_dict[i.parts[-1][0:-4]])[0])
    tmax = int((time_dict[i.parts[-1][0:-4]])[1])
    data = data.filter(l_freq=None,h_freq=30)
    sampling_frequency = 250
    data = data.resample(sfreq=250)
    crop = np.arange(tmin,tmax,5)
    one_trail_data = []
    p = 1
    while(1):
        data_one_trail = data.copy().crop(tmin=tmin,tmax=tmin+5)
        for j in picks_list:
            data_one_channel =  data_one_trail.get_data(units='uV',picks=j)
            one_trail_data.append(data_one_channel)
        one_trail_data = np.asarray(one_trail_data)
        np.save(f'data/test_data/npy/pos/{i.parts[-1][0:-4]}-trail{p}_pos',one_trail_data)
        one_trail_data = []
        tmin = tmin + 5
        if(tmin > (tmax-5)):
            break
        p += 1
        
        
time_vector = np.linspace(0,5,1251)
interpfact = 250 #连续化时的采样率
refractory = 3e-4
dt = 0.01
sample = []
pos_dir = Path('data/test_data/npy/pos')
spike_array_one_trail = []
spike_array = []

def get_spikes(spiketimes,time,dt):
    spikelist = []
    spiketimes = np.around(spiketimes,2)
    spiketimes = spiketimes*100
    spiketimes = spiketimes.astype(np.int16)
    for i in range(int(time/dt)):

        if (i in spiketimes):
            spikelist.append(1)
        else:
            spikelist.append(0)
    return np.asarray(spikelist)


for i in tqdm(sorted(pos_dir.rglob('*.npy')),colour='cyan'):
    array = np.load(str(i))
    array = array.reshape(2,1251)
    for elc in np.arange(2):
        data_one_channel = array[elc]
        ecog_signal, ecog_spikes = pre_process_data(raw_signal = data_one_channel,
                                                time_vector = time_vector, 
                                                interpfact = interpfact, 
                                                refractory = refractory, 
                                                )
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
    np.save(f'data/test_data/spike/pos/{i.parts[-1][0:-4]}_spike',spike_array_one_trail)
    spike_array_one_trail = []

















