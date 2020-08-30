# coding=utf-8
'''
 same 
'''


import torch
from torch.nn import Module
from abc import abstractmethod

import Putil.base.logger as plog
from Putil.data.io_convertor import IOConvertor


class IOConvertorModule(IOConvertor, Module):
    def __init__(self, io):
        Module.__init__(self)
        IOConvertor.__init__(self, io)
        pass

    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args):
        pass