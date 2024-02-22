# coding=utf-8
from Putil.demo.deep_learning.base.indicator import Indicator

class DefaultIndicator(Indicator):
    def __init__(self, args):
        Indicator.__init__(self, args)
        pass

    def forward(self, *input):
        raise NotImplementedError('DefaultIndicator is not implemented')