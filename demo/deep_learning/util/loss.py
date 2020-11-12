# coding=utf-8
from Putil.demo.deep_learning.base.loss import Loss

class DefaultLoss(Loss):
    def __init__(self, args):
        Loss.__init__(self, args)
        pass

    def forward(self, *input):
        raise NotImplementedError('DefaultLoss is not implemented')