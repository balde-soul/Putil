# coding=utf-8
from Putil.demo.deep_learning.base import indicator_statistic


class DefaultIndicatorStatistic(indicator_statistic.IndicatorStatistic):
    def __init__(self, args, property_type='', **kwargs):
        indicator_statistic.IndicatorStatistic.__init__(self, args, property_type, **kwargs)

    def forward(self, *input):
        raise NotImplementedError('DefaultIndicatorStatistic is not implemented')