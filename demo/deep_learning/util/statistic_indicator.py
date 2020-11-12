# coding=utf-8
from Putil.demo.deep_learning.base.statistic_indicator import StatisticIndicator


class DefaultStatisticIndicator(StatisticIndicator):
    def __init__(self, args):
        StatisticIndicator.__init__(self, args)

    def forward(self, *input):
        raise NotImplementedError('DefaultStatisticIndicator is not implemented')