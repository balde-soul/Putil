# coding=utf-8
from Putil.demo.deep_learning.base.model import Model


class DefaultModel(Model):
    def __init__(self, args):
        Model.__init__(self, args)
    
    def forward(self, *input):
        raise NotImplementedError('DefaultModel is not implemented')