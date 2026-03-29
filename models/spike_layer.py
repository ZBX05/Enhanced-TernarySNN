import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from IPython import embed

# record_mem=False
# complementary = False

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True) -> None:
        self._spiking = use_spike

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x



def spike_activation(x:torch.Tensor, binary:bool=False, a:float=1.0, temp:float=1.0) -> torch.Tensor:
    if binary:
        out_s = torch.gt(x, 0.5)
        out_bp = torch.clamp(x, 0, 1)
        #out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        return (out_s.float() - out_bp).detach() + out_bp
    else:
        out_s = torch.sign(x)
        out_s[torch.abs(x)<0.5] = torch.tensor(0.)
        out_bp = torch.clamp(x, -a, a)
        #out_bp[out_bp>0.] = (torch.tanh(temp * (out_bp[out_bp>0.]-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        #out_bp[out_bp<=0.] = (torch.tanh(temp * (out_bp[out_bp<=0.]+0.5)) - np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        return (out_s.float() - out_bp).detach() + out_bp

def MPR(s,thresh):

    s[s>1.] = s[s>1.]**(1.0/3)
    s[s<0.] = -(-(s[s<0.]-1.))**(1.0/3)+1.
    s[(0.<s)&(s<1.)] = 0.5*torch.tanh(3.*(s[(0.<s)&(s<1.)]-thresh))/np.tanh(3.*(thresh))+0.5
    
    return s


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y

def dvs_complementary_mem_update(x_in:torch.Tensor, mem:torch.Tensor, mem_t:torch.Tensor, V_th:float, decay:float, 
                             alpha:nn.Parameter, beta:nn.Parameter, gamma:nn.Parameter,
                             fire_ratio:float, grad_scale:float=1., temp:float=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mem = mem * decay
    # mem_t = torch.sigmoid(alpha) * mem_t.relu() + torch.sigmoid(beta) * -((-mem_t).relu()) + torch.sigmoid(gamma) * mem
    mem_t = torch.sigmoid(alpha) * mem_t + torch.sigmoid(beta) * mem.relu() + torch.sigmoid(gamma) * -((-mem).relu())
    mem = mem_t + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    #mem2 = MPR(mem, 0.5)
    pre_reset_mem = mem
    spike = spike_activation(mem / V_th, a=V_th, temp=temp)
    mem = mem * (1 - torch.abs(spike))
    spike = spike * fire_ratio
    return mem, spike, pre_reset_mem, mem_t

def static_complementary_mem_update(x_in:torch.Tensor, mem:torch.Tensor, mem_t:torch.Tensor, V_th:float, decay:float, 
                             alpha:nn.Parameter, beta:nn.Parameter, gamma:nn.Parameter,
                             fire_ratio:float, grad_scale:float=1., temp:float=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mem = mem * decay
    mem_t = torch.sigmoid(alpha) * mem_t.relu() + torch.sigmoid(beta) * -((-mem_t).relu()) + torch.sigmoid(gamma) * mem
    # mem_t = torch.sigmoid(alpha) * mem_t + torch.sigmoid(beta) * mem.relu() + torch.sigmoid(gamma) * -((-mem).relu())
    mem = mem_t + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    #mem2 = MPR(mem, 0.5)
    pre_reset_mem = mem
    spike = spike_activation(mem / V_th, a=V_th, temp=temp)
    mem = mem * (1 - torch.abs(spike))
    spike = spike * fire_ratio
    return mem, spike, pre_reset_mem, mem_t

def mem_update(x_in:torch.Tensor, mem:torch.Tensor, V_th:float|torch.Tensor, decay:float, fire_ratio:float, 
               grad_scale:float=1., temp:float=1.0) -> tuple[torch.Tensor, torch.Tensor]:
    mem = mem * decay + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    #mem2 = MPR(mem, 0.5)
    pre_reset_mem = mem
    spike = spike_activation(mem / V_th, a=V_th, temp=temp)
    mem = mem * (1 - torch.abs(spike))
    #mem = mem - spike
    spike = spike * fire_ratio
    return mem, spike, pre_reset_mem


class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, step, V_th=0.5, tau=0.25, complementary='static', tmpr=False):
        super(LIFAct, self).__init__()
        self.step = step
        self.V_th = V_th
        self.decay = tau
        self.complementary = complementary
        self.tmpr = tmpr # This variable is used to control the detachment of membrane potential tensor since vanilla LIF does not require the gradient of temporal membrane potential (which is for tmpr method).
        if self.complementary:
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.gamma = nn.Parameter(torch.tensor(0.0))
        # else:
        #     self.alpha = None
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        #self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = 0.1
        
        #self.fire_ratio = nn.Parameter(torch.ones(1,512,1,1), requires_grad=True)
        # self.fire_ratio = nn.Parameter(torch.tensor(1.))
        self.fire_ratio = 1


    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.zeros_like(x[0])
        if self.complementary:
            mem_t = torch.zeros_like(x[0])
        out = []
        self.mem = []
        T, B, C, H, W = x.shape
        for i in range(self.step):
            if self.complementary == 'static':
                u, out_i, mem, mem_t = static_complementary_mem_update(x_in=x[i], mem=u, mem_t=mem_t, V_th=2*self.V_th, decay=self.decay, 
                                                                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                                                                fire_ratio=self.fire_ratio, 
                                                                grad_scale=self.grad_scale, temp=self.temp)
            elif self.complementary == 'dvs':
                u, out_i, mem, mem_t = dvs_complementary_mem_update(x_in=x[i], mem=u, mem_t=mem_t, V_th=2*self.V_th, decay=self.decay, 
                                                                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                                                                fire_ratio=self.fire_ratio, 
                                                                grad_scale=self.grad_scale, temp=self.temp)
            else:
                u, out_i, mem = mem_update(x_in=x[i], mem=u, V_th=2*self.V_th, fire_ratio=self.fire_ratio, 
                                              grad_scale=self.grad_scale, decay=self.decay, temp=self.temp)
            out += [out_i]
            self.mem += [mem if self.tmpr else mem.detach()]
        out = torch.stack(out)
        # self.out = out # For firing rate calculation
        return out


class SpikeConv(SpikeModule):


    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.conv(x)
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out


class SpikePool(SpikeModule):

    def __init__(self, pool, step=2):
        super().__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.pool(x)
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = self.pool(out)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out

class myBatchNorm3d(SpikeModule):
    def __init__(self, BN:nn.BatchNorm2d, step=2):
        super().__init__()
        # self.BN = BN
        self.bn = nn.BatchNorm3d(BN.num_features)
        self.step = step
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self._spiking is not True:
        #     return BN(x)
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out


class tdBatchNorm2d(nn.BatchNorm2d, SpikeModule):
    """Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, bn:nn.BatchNorm2d, alpha:float):
        super(tdBatchNorm2d, self).__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.alpha = alpha
        self.V_th = 0.5
        # self.weight.data = bn.weight.data
        # self.bias.data = bn.bias.data
        # self.running_mean.data = bn.running_mean.data
        # self.running_var.data = bn.running_var.data

    def forward(self, input):
        if self._spiking is not True:
            # compulsory eval mode for normal bn
            self.training = False
            return super().forward(input)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        channel_dim = input.shape[2]
        input = self.alpha * self.V_th * (input - mean.reshape(1, 1, channel_dim, 1, 1)) / \
                (torch.sqrt(var.reshape(1, 1, channel_dim, 1, 1) + self.eps))
        if self.affine:
            input = input * self.weight.reshape(1, 1, channel_dim, 1, 1) + self.bias.reshape(1, 1, channel_dim, 1, 1)

        return input
