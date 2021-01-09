# coding=utf-8
import torch
from torch.optim import Adam
from torch.optim import Adadelta
from torch.nn import Module

from Putil.torch.optimization.combine_optimization import CombineOptimization


class Multi(Module):
    def __init__(self):
        Module.__init__(self)
        self._a = torch.Tensor([0.1])
        self._a.requires_grad = True
        self._b = torch.Tensor([0.5])
        self._b.requires_grad = True

    def forward(self):
        return self._a * self._b
    pass

class Loss(Module):
    def __init__(self):
        Module.__init__(self)
        self._c = torch.Tensor([0.1])
        self._c.requires_grad = True
        pass

    def forward(self, x):
        return x * self._c - 1.0
    pass

multi = Multi()
loss = Loss()

opt1 = Adam(multi.parameters())
opt2 = Adadelta(loss.parameters())

l = loss(multi())
l.backward()

opt1.step()
opt2.step()