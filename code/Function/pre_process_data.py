import scipy as sc
import numpy as np
from scipy.signal import butter, lfilter,  filtfilt
import matplotlib
import matplotlib.pyplot as plt

#========================================================================================
# Preprocessing ECoG signals
#========================================================================================   
'''
This function performs the preprocessing stages to find HFO in ECoG recordings:
Filtering: the wideband signal is filtered between 250-500 Hz
Baseline detection: the background noise of the signal is used to set the threshold for the signal to spike conversion.
Signal to spike conversion: the clocked signal is converted into asynchronous spikes.

'''
def pre_process_data(raw_signal,time_vector, interpfact, refractory):


    Spiking_threshold = 40

    FR_up, FR_dn  = signal_to_spike_refractory(interpfact = interpfact, 
                                           time = time_vector,
                                           signal = raw_signal ,
                                           thr_up = Spiking_threshold, thr_dn = Spiking_threshold, 
                                           refractory = refractory)

    FR_up = np.asarray(FR_up)
    FR_dn = np.asarray(FR_dn)


    Signal = {}
    Signal['Ecog'] = raw_signal
    Signal['time'] = time_vector
    

    Spikes = {}
    Spikes['threshold'] = np.asarray(Spiking_threshold)
    Spikes['up'] = FR_up
    Spikes['dn'] = FR_dn

    return Signal, Spikes




#========================================================================================
# Butterworth filter coefficients
#========================================================================================   
'''
These functions are used to generate the coefficients for lowpass, highpass and bandpass
filtering for Butterworth filters.

:cutOff (int): either the lowpass or highpass cutoff frequency
:lowcut, highcut (int): cutoff frequencies for the bandpass filter
:fs (float): sampling frequency of the wideband signal
:order (int): filter order 
:return b, a (float): filtering coefficients that will be applied on the wideband signal

'''
#def butter_lowpass(cutOff, fs, order=5):
#    nyq = 0.5 * fs
#    #normalCutoff = cutOff / nyq
#    normalCutoff = 2 * cutOff / fs
#    b, a = butter(order, normalCutoff, btype='low', analog = True)
#    return b, a

def butter_lowpass(cutOff, fs, order=5):
    normalCutoff = 2 * cutOff / fs
    b, a = butter(order, normalCutoff)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#========================================================================================
# Butterworth filters
#========================================================================================   
'''
These functions apply the filtering coefficients calculated above to the wideband signal.

:data (array): vector with the signal values of the wideband signal 
:cutOff (int): either the lowpass or highpass cutoff frequency
:lowcut, highcut (int): cutoff frequencies for the bandpass filter
:fs (float): sampling frequency of the wideband signal
:order (int): filter order 
:return y (array): vector with signal of the filtered signal

'''
def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#========================================================================================
# Threshold calculation based on the noise floor
#========================================================================================   
'''
This functions retuns the mean threshold for your signal, based on the calculated 
mean noise floor and a user-specified scaling facotr that depeneds on the type of signal,
characteristics of patterns, etc.

:signal (array): signal of the signal
:time (array): time vector
:window (float): time window [same units as time vector] where the maximum signal of the signal 
                 will be calculated
:chosen_samples (int): from the maximum values in each window time, only these number of
                       samples will be used to calculate the mean maximum signal.
: scaling_factr (float): a percentage of the calculated threshold

'''

def find_thresholds(signal, time, window, step_size, chosen_samples, scaling_factor ):
    window_size = window
    trial_duration =np.max(time)
    num_timesteps = int(np.ceil(trial_duration / step_size))
    max_min_signal = np.zeros((num_timesteps, 2))
    
    #找到每一个窗中的最小值和最大值并放在一个矩阵中~
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=trial_duration,step=step_size)):        
        interval=[interval_start, interval_start + window_size]
        start_time, end_time = interval
        index = np.where(np.logical_and(time >= start_time, time <= end_time))[0]
        signal = signal.reshape(-1)
        max_signal = np.max(signal[index])
        min_signal = np.min(signal[index])
        max_min_signal[interval_nr,0] = max_signal
        max_min_signal[interval_nr,1] = min_signal  

    threshold_up = np.mean(np.sort(max_min_signal[:,0])[:chosen_samples])
    threshold_dn = np.mean(np.sort(max_min_signal[:,1] * -1)[:chosen_samples])
    mean_threshold = scaling_factor*(threshold_up + threshold_dn)
    
    return mean_threshold

#========================================================================================
# Signal to spike conversion with refractory period
#========================================================================================   
'''
This functions retuns two spike trains, when the signal crosses the specified threshold in 
a rising direction (UP spikes) and when it crosses the specified threshold in a falling 
direction (DOWN spikes)

:time (array): time vector
:signal (array): signal of the signal
:interpfact (int): upsampling factor, new sampling frequency
:thr_up (float): threshold crossing in a rising direction
:thr_dn (float): threshold crossing in a falling direction
:refractory (float): period in which no spike will be generated [same units as time vector]
'''

def signal_to_spike_refractory(interpfact, time, signal, thr_up, thr_dn,refractory):
    actual_dc = 0 
    spike_up = []
    spike_dn = []

    f = sc.interpolate.interp1d(time, signal)        #Continuousize a discrete signal    
    rangeint = np.round((np.max(time) - np.min(time))*interpfact)
    xnew = np.linspace(np.min(time), np.max(time), num=int(rangeint), endpoint=True) #Re-segment sample points
    data = np.reshape([xnew, f(xnew)], (2, len(xnew))).T  

    i = 0
    while (i < (len(data))):
        if( (actual_dc + thr_up) < data[i,1]):
            spike_up.append(data[i,0] )  #spike up
            actual_dc = data[i,1]        # update current dc value
            i += int(refractory * interpfact)
        elif( (actual_dc - thr_dn) > data[i,1]):
            spike_dn.append(data[i,0] )  #spike dn
            actual_dc = data[i,1]        # update curre
            i += int(refractory * interpfact)
        else:
            i += 1

    return spike_up, spike_dn
    
