import sinabs.layers as sl
import torch
import torch.nn as nn

def detach_states(snn: nn.Module):
    
    for lyr in snn.modules():
        
        if isinstance(lyr, sl.StatefulLayer):
            
            for name, buffer in lyr.named_buffers():
         
                buffer.detach_()
            