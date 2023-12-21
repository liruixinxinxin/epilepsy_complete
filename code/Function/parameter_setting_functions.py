import torch
import numpy as np
from rockpool import TSContinuous, TSEvent

def set_weights_core_SNN(weight_matrix_shape, weight_vector, num_neurons):
    
    half_weights = int(weight_vector.numel()/2)
    # Create new weight matrix
    new_weight_matrix = torch.zeros(weight_matrix_shape)
    #Skip the weights that make use of the same time constant
    j = 0
    for i in range(num_neurons):
        new_weight_matrix[0,j] = weight_vector[:half_weights:][i]
        new_weight_matrix[1,j+1] = weight_vector[half_weights:][i]
        j+=2
    
    
    return torch.nn.Parameter(new_weight_matrix, 
                                   requires_grad  = False)



def get_input_spikes(spiketimes, neuronID, dt,t_start, t_stop, num_timesteps):

    asynchronous_input = TSEvent(spiketimes, neuronID, 
                             t_start = t_start, t_stop = t_stop)

    synchronous_input = asynchronous_input.raster(dt = dt,add_events = True,num_timesteps =num_timesteps)

    
    synchronous_input = torch.from_numpy(synchronous_input)*1
    synchronous_input = synchronous_input.to(torch.float32)
    synchronous_input = synchronous_input.reshape(1,synchronous_input.shape[0],synchronous_input.shape[1])
    synchronous_input = synchronous_input[:,:num_timesteps,:]
    
    
    return asynchronous_input, synchronous_input
