import math
from PSN.cifar10dvs.spikingjelly.clock_driven import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor
from typing import Callable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x

class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(10, 1, 1, 1, 1, device=device))
    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y

class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

# class ZIF(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, v_th, gama):
#         out = torch.sign(input / v_th)
#         out[torch.abs(input / v_th)<1.0] = torch.tensor(0.)
#         L = torch.tensor([gama])
#         V = torch.tensor([v_th])
#         ctx.save_for_backward(input, out, L, V)
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input, out, L, V) = ctx.saved_tensors
#         gama = L[0].item()
#         v_th = V[0].item()
#         grad_input = grad_output.clone()
#         tmp = (1 / gama) * (1 / gama) * ((gama - ((input-v_th).abs())).clamp(min=0))
#         grad_input = grad_input * tmp
#         # tmp = (torch.abs(input) <= gama).float()
#         # grad_input = grad_input * tmp
#         return grad_input, None, None

def spike_activation(x:torch.Tensor) -> torch.Tensor:
    out_s = torch.sign(x)
    out_s[torch.abs(x)<0.5] = torch.tensor(0.)
    out_bp = torch.clamp(x, -1, 1)
    #out_bp[out_bp>0.] = (torch.tanh(temp * (out_bp[out_bp>0.]-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    #out_bp[out_bp<=0.] = (torch.tanh(temp * (out_bp[out_bp<=0.]+0.5)) - np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp

class ComplementaryTernarySpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(ComplementaryTernarySpike, self).__init__()
        self.heaviside = spike_activation
        self.v_th = thresh
        self.tau = tau
        self.sg_gamma = gamma
        self.pre_spike_mem = []
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mem_v = []
        _mem = []
        mem = torch.zeros_like(x[:, 0, ...])
        mem_t = 0
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau
            mem_t = torch.sigmoid(self.alpha) * mem_t + torch.sigmoid(self.beta) * mem.relu() + torch.sigmoid(self.gamma) * -((-mem).relu())
            mem = mem_t + x[:, t, ...]
            _mem.append(mem)
            spike = self.heaviside(mem / self.v_th)
            mem = mem * (1 - torch.abs(spike))
            mem_v.append(spike)
        self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.pre_spike_mem = []

    def forward(self, x):
        mem_v = []
        _mem = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            _mem.append(mem.detach().cpu().clone())
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)
        self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)

class VGGSNN(nn.Module):
    def __init__(self, tau=0.25):
        super(VGGSNN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            Layer(64, 128, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            pool,
            Layer(128, 256, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            Layer(256, 256, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            pool,
            Layer(256, 512, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            Layer(512, 512, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            pool,
            Layer(512, 512, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            Layer(512, 512, 3, 1, 1),
            ComplementaryTernarySpike(tau=self.tau),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x