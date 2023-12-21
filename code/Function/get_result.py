
import numpy as np
from tqdm.auto import tqdm
import torch
from Function.pre_process_data import *
from Function.Spike_manager_functions import *
from Function.parameter_setting_functions import *
from tqdm.auto import tqdm 

def get_result(time_vector,interpfact,refractory,net):
    global array
    global n_windows
    global spike_array_one_trail
    global signal_start
    global dt
    sim = net
    signal_start = signal_start + 125
    array_slices = array[:,signal_start:signal_start+1250]
    for elc in np.arange(2):
        data_one_channel = array_slices[elc]
        print(data_one_channel.shape)
        print(time_vector.shape)
        print(signal_start)
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
            dt_original_data = 1/250
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
    tensor = torch.from_numpy(spike_array_one_trail.T)
    data = torch.tensor(tensor,dtype=torch.float)
    data = torch.reshape(data,(500,4))
    data = data.numpy()
    data = data.astype(int)
    sim.reset_state()
    out, _, recordings = sim((data*3).clip(0, 15),record=True,record_power = True,read_timeout = 20)
    out = recordings['Isyn_out'].squeeze()
    peaks = out.max(0)
    result = peaks.argmax()
    spike_array_one_trail = []
    print(result.item())
    return result.item()
    
