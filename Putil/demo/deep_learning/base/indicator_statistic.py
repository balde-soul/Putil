'''
 @note
 此文件提供一个统计性的指标计算方法，在evaluate之后，判断是否提升，如果提升则需要计算统计性指标
'''
# coding=utf-8
from abc import abstractmethod
from torch.nn import Module
import copy


class IndicatorStatistic:
    '''
     @brief
     @note 接收训练阶段的train或者evaluate每个step的decode输出，再最后一个step输出统计指标
    '''
    def __init__(self, args, property_type='', **kwargs):
        pass

    @abstractmethod
    def add_step_output(self, *input):
        '''
         @brief
         @note 每个step的decode输出传入到该函数，进行统计处理
        '''
        pass

    @abstractmethod
    def statistic_out(self):
        '''
         @brief
         @note 最后一个step运行完，decode输出传到add_step_output之后就可以调用此函数进行统计，输出一个具有<比较函数>实现的对象
        '''
        pass
    pass


class _DefaultIndicatorStatistic(IndicatorStatistic):
    def __init__(self, args, property_type='', **kwargs):
        IndicatorStatistic.__init__(self, args, property_type, **kwargs)
        pass
    pass


def DefaultIndicatorStatistic(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_default_indicator_statistic():
        return _DefaultIndicatorStatistic(temp_args, property_type, **kwargs)
    return generate_default_indicator_statistic


def DefaultIndicatorStatisticArg(parser, property_type='', **kwargs):
    pass