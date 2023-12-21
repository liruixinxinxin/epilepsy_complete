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
  'chb01_03': [2996, 3036], 'chb01_04': [1467, 1494], 'chb01_15': [1732, 1772], 'chb01_16': [1015, 1066], 
 'chb01_18': [1720, 1810], 'chb01_21': [327, 420], 'chb01_26': [1862, 1963], 
 'chb02_16': [2972, 3053], 'chb02_19': [3369, 3378], 
 'chb03_01': [372, 414], 'chb03_02': [741, 796], 'chb03_03': [437, 501], 'chb03_04': [2180, 2214],
 'chb03_34': [1982, 2029], 'chb03_35': [2592, 2656], 'chb03_36': [1725, 1778], 
 'chb04_05': [7804, 7853], 'chb03_35': [2592, 2656], 'chb03_36': [1725, 1778], 
 'chb05_06': [417, 532], 'chb05_13': [1086, 1196], 'chb05_16': [2317, 2413], 'chb05_17': [2451, 2517], 
 'chb05_22': [2348, 2465], 
 'chb06_01': [1724, 1738], 'chb06_04': [6211, 6231], 'chb06_09': [12500, 12516], 
 'chb06_10': [10833, 10845], 'chb06_13': [506, 519], 'chb06_18': [7799, 7811], 'chb06_24': [9387, 9403], 
 'chb07_12': [4920, 5006], 'chb07_13': [3285, 3381], 'chb07_19': [13688, 13816], 
 'chb08_02': [2670, 2841], 'chb08_05': [2856, 3036], 'chb08_11': [2988, 3122], 'chb08_13': [2417, 2572], 
 'chb08_21': [2090, 2347], 
 'chb09_06': [12231, 12295], 'chb09_08': [2951, 3030], 'chb09_19': [5299, 5361], 
 'chb10_12': [6313, 6348], 'chb10_20': [6888, 6958], 'chb10_27': [2382, 2447], 'chb10_30': [3021, 3079], 
 'chb10_31': [3801, 3877], 'chb10_38': [4618, 4707], 'chb10_89': [1383, 1437], 
 'chb11_82': [301,320],  'chb11_99': [1464, 2206], 
 'chb12_06': [1670, 1726],'chb12_23': [645, 670], 
 'chb12_42': [725, 750], 'chb12_33': [2190, 2206],
 'chb13_19': [2077, 2121], 'chb13_21': [934, 1004],'chb13_40': [530, 594],'chb13_55': [2436, 2454],
 'chb13_59': [3339, 3401], 'chb13_62': [2664, 2721],
 'chb14_03': [1986, 2000], 'chb14_04': [1372, 1392], 'chb14_06': [1911, 1925], 'chb14_11': [1838, 1879], 
 'chb14_17': [3239, 3259], 'chb14_18': [1039, 1061], 'chb15_06': [300, 397], 
 'chb15_10': [1082, 1113], 'chb15_15': [1591, 1748], 'chb15_17': [1920, 1960],  'chb15_20': [607, 662], 
 'chb15_22': [760, 965], 'chb15_28': [876, 1066],'chb15_40': [834, 894], 'chb15_46': [3322, 3429], 
 'chb15_49': [1108, 1248], 'chb15_52': [778, 849], 'chb15_54': [843, 10220], 'chb15_62': [751, 895],
 'chb16_10': [2290, 2299], 'chb16_11': [1120,1129],'chb16_14': [1854, 1868], 'chb16_18': [627, 635],
 'chb18_29': [3477, 3527], 'chb18_30': [541,571],'chb18_31': [2087, 2155], 'chb18_32': [1920, 1960],
 'chb18_35': [2228, 2250], 'chb18_36': [463,509],
 'chb19_28': [319,377],'chb19_29': [2970, 3041], 'chb19_30': [3179, 3240],
 'chb20_12': [94, 123],'chb20_13': [2498, 2537], 'chb20_16': [2226, 2261], 'chb20_68': [1405, 1432],  
 'chb21_19': [1288,1344],'chb21_20': [2627, 2677], 'chb21_21': [2003, 2084],
 'chb21_22': [2553, 2565],
 'chb22_20': [3367, 3425], 'chb22_25': [3139, 3213], 'chb22_38': [1263, 1335], 
 'chb23_06': [3975, 4075], 'chb23_08': [5104, 5151],'chb23_09': [2610, 2660], 
 'chb24_01': [2451, 2470], 'chb24_03': [2883, 2908], 'chb24_04': [1088, 1122], 'chb24_06': [1229, 1245], 
 'chb24_07': [38, 60],'chb24_09': [1745, 1760],'chb24_11': [3527, 3597],'chb24_13': [3288, 3304],'chb24_14': [1939, 1955],
 'chb24_15': [3552, 3569],'chb24_17': [3515, 3581],'chb24_21': [2804, 2872],
 
 }

dir = Path('raw_selected/raw_selected/train')
for i in tqdm(sorted(dir.rglob('*.edf'))):
    data = mne.io.read_raw_edf(str(i),preload=True)
    tmin = int((time_dict[i.parts[-1][0:-4]])[0])
    tmax = int((time_dict[i.parts[-1][0:-4]])[1])
    sampling_frequency = 250
    data = data.filter(l_freq=None,h_freq=30)
    data = data.resample(sfreq=sampling_frequency)
    one_trail_data = []
    p = 1
    while(1):
        data_one_trail = data.copy().crop(tmin=tmin,tmax=tmin+5)
        for j in picks_list:
            data_one_channel =  data_one_trail.get_data(units='uV',picks=j)
            one_trail_data.append(data_one_channel)
        one_trail_data = np.asarray(one_trail_data)
        np.save(f'data/train_data/npy/pos/{i.parts[-1][0:-4]}-trail{p}_pos',one_trail_data)
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
pos_dir = Path('data/train_data/npy/pos')
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
    np.save(f'data/train_data/spike/pos/{i.parts[-1][0:-4]}_spike',spike_array_one_trail)
    spike_array_one_trail = []

















