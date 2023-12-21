
import numpy as np

'''
This set of functions consist of a function that wraps all the spikes from different channels
into vector of spike times and neuron ID (similar to the vectors returned by DYNAPSE)
and a mean firing rate giving a vector of spiketimes and neuron ID with configurable
window and step size for computing the average activity.

'''

def concatenate_spikes(spikes_list):

    '''
    Get spikes per channel in a dictionary and concatenate them in one ingle vector with 
    spike times and neuron ids.
    :param spikes_list (dict): dict where the key is the channel name and contains a vector
                               with spike times 
    :return all_spiketimes (array): vector of all spike times
    :return all_neuron_ids (array): vector of all neuron ids 
    '''

    all_spiketimes = []
    all_neuron_ids = []
    channel_nr = 0
    for key in spikes_list:
        if channel_nr == 0:
            all_spiketimes = spikes_list['%s' %key]
            all_neuron_ids = np.ones_like(all_spiketimes) * channel_nr
            channel_nr +=1
        else:
            new_spiketimes = spikes_list['%s' %key]
            all_spiketimes = np.concatenate((all_spiketimes,new_spiketimes), axis=0)
            all_neuron_ids = np.concatenate((all_neuron_ids,
                                             np.ones_like(new_spiketimes) * channel_nr), axis=0)
            channel_nr +=1

    sorted_index = np.argsort(all_spiketimes)
    all_spiketimes_new = all_spiketimes[sorted_index]
    all_neuron_ids_new = all_neuron_ids[sorted_index]
    return all_spiketimes_new, all_neuron_ids_new

    

def get_spikes(spiketimes,tmin,tmax,dt):
    spikelist = []
    spiketimes = np.around(spiketimes,2)
    spiketimes = spiketimes*100
    spiketimes = spiketimes.astype(np.int32)
    for i in range(tmin*100,tmax*100):
        if (i in spiketimes):
            spikelist.append(1)
        else:
            spikelist.append(0)
    return np.asarray(spikelist)
