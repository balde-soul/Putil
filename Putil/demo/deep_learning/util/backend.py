# coding=utf-8
from Putil.demo.deep_learning.base.backend import Backend

class DefaultBackend(Backend):
    def __init__(self, args):
        Backend.__init__(self, args)

    def forward(self, x):
        raise NotImplementedError('foward of BackendNormal is not implemented')