import tkinter as tk
import mne
import numpy as np
import torch
import matplotlib.image as mpimg

from tqdm.auto import tqdm
from Function.pre_process_data import *
from Function.Spike_manager_functions import *
from Function.parameter_setting_functions import *
from tqdm.auto import tqdm 
from Function.get_result import *
from matplotlib.patches import Circle, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
img = mpimg.imread('epilespy.png')


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
 
 
 'chb20_12': [104, 123],'chb20_13': [2498, 2537], 'chb20_16': [2226, 2261], 'chb20_68': [1405, 1432],  
 
 'chb21_19': [1288,1344],'chb21_20': [2627, 2677], 'chb21_21': [2003, 2084],
 'chb21_22': [2553, 2565],
  
 'chb22_20': [3367, 3425], 'chb22_25': [3139, 3213], 'chb22_38': [1263, 1335], 
  
 'chb23_06': [3975, 4075], 'chb23_08': [5104, 5151],'chb23_09': [2610, 2660], 
 
 'chb24_01': [2451, 2470], 'chb24_03': [2883, 2908], 'chb24_04': [1088, 1122], 'chb24_06': [1229, 1245], 
 'chb24_07': [38, 60],'chb24_09': [1745, 1760],'chb24_11': [3567, 3597],'chb24_13': [3288, 3304],'chb24_14': [1939, 1955],
 'chb24_15': [3552, 3569],'chb24_17': [3515, 3581],'chb24_21': [2804, 2872],
 }

#===============================================================#
#定义网络和Xylosim#
#===============================================================#
# - Numpy
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
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
# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.devices import xylo as x
sr = 250
red_light = Circle(xy=(0.2, 0.5), radius=0.1, color='red')
yellow_light = Circle(xy=(0.5, 0.5), radius=0.1, color='orange')
green_light = Circle(xy=(0.8, 0.5), radius=0.1, color='green')
frame = Rectangle(xy=(0, 0), width=1, height=1, fill=False)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
net = WaveSenseNet(
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
net.load('/home/ruixing/workspace/chbtar/chb/models/SNN_model_Isyn.pth')


g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.vA2.config_from_specification(**spec)
sim = x.vA2.XyloSim.from_config(config)
#==============================================================#
#完成#
#==============================================================#
sampling_frequency = 250
file = "chb01_04.edf"
try:
    data = mne.io.read_raw_edf(f'raw_selected/raw_selected/train/{file}',preload=True)
except:data = mne.io.read_raw_edf(f'raw_selected/raw_selected/test/{file}',preload=True)
data = data.filter(l_freq=None,h_freq=30)
data = data.resample(sfreq=sampling_frequency)
tmin_seizure = int(time_dict[(file[0:8])][0])
tmax_seizure = int(time_dict[(file[0:8])][1])
tmin = tmin_seizure - 10
tmax = tmax_seizure + 10
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
number_of_button = 1
for j in picks_list:
    data_one_channel = data.get_data(units='uV',picks=j)
    one_sample_data.append(data_one_channel)
one_sample_data = np.asarray(one_sample_data)
array = one_sample_data.reshape(2,-1)



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
    output, _, recordings = sim((data*2.8).clip(0, 15),record=True,record_power = True,read_timeout = 20)
    out = recordings['Isyn_out'].squeeze()
    peaks = out.max(0)
    result = peaks.argmax()
    spike_array_one_trail = []
    return result.item(),synchronous_input,output
def detect_seizure(result_list,result):
    global normal
    global siezure
    if normal == True:
        if result == 0:
            ax3.text(0.5, 0.5, "Result: no epilespy", va='center', ha='center', fontsize=20)
            ax3.axis('off')
            red_light.set_visible(False)
            yellow_light.set_visible(False)
            green_light.set_visible(True)
            ax_light.axis('off')
        if result == 1:
            if result_list[-3:] != [1, 1, 1]:
                ax3.text(0.5, 0.5, "Result: suspected epilespy", va='center', ha='center', fontsize=20)
                ax3.axis('off')
                red_light.set_visible(False)
                yellow_light.set_visible(True)
                green_light.set_visible(False)
                ax_light.axis('off')
            if result_list[-3:] == [1, 1, 1]: 
                red_light.set_visible(True)
                yellow_light.set_visible(False)
                green_light.set_visible(False)
                ax_light.axis('off')
                ax3.text(0.5, 0.5, "Result: found epilespy", va='center', ha='center', fontsize=20)
                ax3.axis('off')
                ax_img.imshow(img)
                ax_img.axis('off') 
                siezure = True
                normal = False
    
    if siezure == True:
        if result == 0:
            if result_list[-3:] != [0, 0, 0]:
                ax3.text(0.5, 0.5, "Result: found epilespy", va='center', ha='center', fontsize=20)
                ax3.axis('off')
                ax_img.imshow(img)
                ax_img.axis('off')
                red_light.set_visible(True)
                yellow_light.set_visible(False)
                green_light.set_visible(False)
                ax_light.axis('off')
                 
            if result_list[-3:] == [0, 0, 0]:
                ax3.text(0.5, 0.5, "Result: end of seizure", va='center', ha='center', fontsize=20) 
                ax3.axis('off')
                siezure = False
                normal =True
                red_light.set_visible(False)
                yellow_light.set_visible(False)
                green_light.set_visible(True)
                ax_light.axis('off')
        if result ==1:
                ax3.text(0.5, 0.5, "Result: found epilespy", va='center', ha='center', fontsize=20)
                ax3.axis('off')
                ax_img.imshow(img)
                ax_img.axis('off')
                red_light.set_visible(True)
                yellow_light.set_visible(False)
                green_light.set_visible(False) 
                ax_light.axis('off')
def move_rect():
    global n
    ax1.patches.clear()
    ax2.patches.clear()

    ax1.axvspan(tmin_seizure, tmax_seizure, alpha=0.5, color='red')
    ax2.axvspan(tmin_seizure, tmax_seizure, alpha=0.5, color='red')
    ax1.axvspan(0+tmin+n*0.5 ,5+tmin+n*0.5, alpha=0.5, color='green')
    ax2.axvspan(0+tmin+n*0.5 ,5+tmin+n*0.5, alpha=0.5, color='green')

    canvas.draw()
    n += 1
    plt.pause(0.1)
def move_and_get():
    global number_of_button
    time = number_of_button*0.5+tmin
    number_of_button += 1
    move_rect()
    global siezure_trigger
    result,spike,output = get_result(time_vector, interpfact, refractory, sim)
    result_list.append(result)
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    ax_img.clear()
    ax_img.axis('off')
    detect_seizure(result_list=result_list,result=result)
    ax4.plot(spike[0])
    ax4.set_ylim([0, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.text(-0.1, 0.5, "up", va='center', ha='center', rotation=90, transform=ax4.transAxes)
    ax5.plot(spike[1])
    ax5.set_ylim([0, 2])
    ax5.set_yticks([0, 1, 2])
    ax5.text(-0.1, 0.5, "down", va='center', ha='center', rotation=90, transform=ax5.transAxes)
    ax6.plot(output.T[0])
    ax7.plot(output.T[1])
    ax6.set_yticks([0, 1, 2])
    ax6.text(-0.1, 0.5, "output[0]", va='center', ha='center', rotation=90, transform=ax6.transAxes)
    ax7.set_yticks([0, 1, 2])
    ax7.text(-0.1, 0.5, "output[1]", va='center', ha='center', rotation=90, transform=ax7.transAxes)
    canvas.draw()
    if 5+time == tmax:
        root.destroy()
def auto_press():
    # 模拟点击按钮
    button2.invoke()
    # 0.5 秒后调用自身
    root.after(20, auto_press)



# 创建一个Tkinter窗口
root = tk.Tk()
root.title("信号数据展示")

# 创建一个Matplotlib图形对象，并创建两个子图对象
fig = Figure(figsize=(20, 8), dpi=100)
ax1 = fig.add_subplot(4,2,1)
ax1.axvspan(tmin_seizure, tmax_seizure, alpha=0.5, color='red')
ax2 = fig.add_subplot(4,2,3)
ax2.axvspan(tmin_seizure, tmax_seizure, alpha=0.5, color='red')
ax3 = fig.add_subplot(4,2,5)
ax3.text(0.5, 0.5, "Result: None", va='center', ha='center', fontsize=20)
ax3.set_axis_off()
ax3.axis('off')
ax4 = fig.add_subplot(4,2,2)
ax5 = fig.add_subplot(4,2,4)
ax6 = fig.add_subplot(4,2,6)
ax7 = fig.add_subplot(4,2,8)
ax_img = fig.add_subplot(4,2,7)
ax_img.set_position([0.1, 0.1, 0.2, 0.2])
ax_img.axis('off')
ax_light = fig.add_subplot(4,2,7)
ax_light.set_aspect('equal')
ax_light.add_patch(red_light)
ax_light.add_patch(yellow_light)
ax_light.add_patch(green_light)
ax_light.add_patch(frame)
ax_light.axis('off')
x = np.linspace(tmin,tmax,array[0].shape[0])
# 在子图1中绘制信号数据data，并在左侧添加标题文本框
ax1.plot(x,array[0])
ax1.text(-0.1, 0.5, "C3-P3", va='center', ha='center', rotation=90, transform=ax1.transAxes)
ax1.axvspan(0+tmin ,5+tmin, alpha=0.5, color='green')
# 在子图2中绘制信号数据array2，并在左侧添加标题文本框
ax2.plot(x,array[1])
ax2.text(-0.1, 0.5, "C4-P4", va='center', ha='center', rotation=90, transform=ax2.transAxes)
ax2.axvspan(0+tmin ,5+tmin, alpha=0.5, color='green')
# 创建一个Tkinter画布对象，并将Matplotlib图形对象绑定到画布上
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()
ax1.plot(x,array[0],color='#3399FF',linewidth=1)
ax2.plot(x,array[1],color='#3399FF',linewidth=1)
# 创建一个按钮，并将按钮与回调函数绑定
time_vector = np.linspace(0,5,1250)
interpfact = 250 #连续化时的采样率
refractory = 3e-4
n = 1
n_windows = 0
spike_array_one_trail=[]
result_list = [0,0,0]
signal_start = 0
siezure_trigger = 0
normal = True
siezure = False
button2 = tk.Button(master=root, text="生成预测值", command=move_and_get)
button2.pack()
auto_press()

tk.mainloop()