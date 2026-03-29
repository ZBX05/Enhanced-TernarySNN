import torch.nn as nn
from models.spike_layer import SpikeConv, LIFAct, tdBatchNorm2d, SpikePool, SpikeModule, myBatchNorm3d
from models.spike_block import specials, SEWBasicBlock


class SpikeModel(SpikeModule):

    def __init__(self, model: nn.Module, step=2, V_th=0.5, tau=0.25, complementary='static', tmpr=False, connect_f='ADD'):
        super().__init__()
        self.model = model
        self.step = step
        self.connect_f = connect_f
        self.V_th = V_th
        self.tau = tau
        lif_to_if = False
        for m in self.model.modules():
            if isinstance(m, SEWBasicBlock):
                lif_to_if = True
                break
        self.spike_module_refactor(self.model, step=step, complementary=complementary, tmpr=tmpr, connect_f=connect_f)
        if lif_to_if:
            for m in self.model.modules():
                if isinstance(m, LIFAct):
                    m.decay = 1.0

    def spike_module_refactor(self, module: nn.Module, step=2, complementary='static', tmpr=False, connect_f='ADD'):
        """
        Recursively replace the normal conv2d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():
            if type(child_module) in specials and type(child_module) is SEWBasicBlock:
                setattr(module, name, specials[type(child_module)](child_module, step=step, complementary=complementary, tmpr=tmpr, connect_f=connect_f))
            elif type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, step=step, complementary=complementary, tmpr=tmpr))

            elif isinstance(child_module, nn.Sequential):
                self.spike_module_refactor(child_module, step=step, complementary=complementary, tmpr=tmpr)

            elif isinstance(child_module, nn.Conv2d):
                setattr(module, name, SpikePool(child_module, step=step))

            elif isinstance(child_module, nn.Linear):
                setattr(module, name, SpikeConv(child_module, step=step))

            elif isinstance(child_module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                setattr(module, name, SpikePool(child_module, step=step))

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                setattr(module, name, LIFAct(step=step, V_th=self.V_th, tau=self.tau, complementary=complementary, tmpr=tmpr))
            elif isinstance(child_module, nn.BatchNorm2d):
                setattr(module, name, SpikeConv(child_module, step=step))
            #elif isinstance(child_module, nn.BatchNorm2d):
            #    setattr(module, name, tdBatchNorm2d(bn=child_module, alpha=1))
            #elif isinstance(child_module, nn.BatchNorm2d):
            #    setattr(module, name, myBatchNorm3d(child_module, step=step))
            
            else:
                self.spike_module_refactor(child_module, step=step, complementary=complementary, tmpr=tmpr)

    def forward(self, input, is_adain=False,is_drop=False):
        
        if len(input.shape) == 4:
            input = input.repeat(self.step, 1, 1, 1, 1)
        else:
            input = input.permute([1, 0, 2, 3, 4])
            
        if is_adain and is_drop:
            fea, out = self.model(input,is_adain=True,is_drop=True)
        elif is_adain and not is_drop:
            fea, out = self.model(input,is_adain=True,is_drop=False)
        elif not is_adain and is_drop:  
            out = self.model(input,is_adain=False, is_drop=True)
        else:
            out = self.model(input,is_adain=False, is_drop=False)
        if len(out.shape) == 3:
            out = out.mean([0])
        if is_adain:
            return fea,out
        else:
            return out        

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(use_spike)

    def set_spike_before(self, name):
        self.set_spike_state(False)
        for n, m in self.model.named_modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(True)
            if name == n:
                break


# from models.resnet import resnet20_cifar_modified
# model = SpikeModel(resnet20_cifar_modified())
# model.set_spike_before('layer1')
# for n, m in model.named_modules():
#     if isinstance(m, SpikeModule):
#         if m._spiking is True:
#             print(n)
# import torch
# model(torch.randn(1,3,32,32))
