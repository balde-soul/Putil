'''
 @note
 本文件提供指标计算方法，该指标应用于train与evaluate中的量化，通过该指标判断train与evaluate的基本情况
 以及进行evaluate是否提升的判断
'''
# coding=utf-8
from abc import abstractmethod
from torch.nn import Module
import copy


class Indicator:
    '''
     @brief
     @note 每个epoch一个轮回，接收train阶段evaluate的decode输出，生成代表性指标，指导lr_reduce、auto_save、auto_stop等
    '''
    def __init__(self, args):
        self._indicator_name = args.indicator_name
        self._indicator_source = args.indicator_source
        pass

    @abstractmethod
    def __call__(self, input):
        '''
         @brief
         @note
         @param[in] input contain the ground truth and the prediction
         @ret return a dict,{str: value}, the value can be reduce
        '''
        pass
    pass


class _DefaultIndicator(Indicator, Module):
    def __init__(self, args):
        Indicator.__init__(self, args)
        Module.__init__(self)
        pass
    pass


def DefaultIndicator(args):
    temp_args = copy.deepcopy(args)
    def generate_default_indicator():
        return _DefaultIndicator(temp_args)
    return generate_default_indicator


def DefaultIndicatorArg(parser):
    pass