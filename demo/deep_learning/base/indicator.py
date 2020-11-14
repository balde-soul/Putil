'''
 @note
 本文件提供指标计算方法，该指标应用于train与evaluate中的量化，通过该指标判断train与evaluate的基本情况
 以及进行evaluate是否提升的判断
'''
# coding=utf-8
from abc import abstractmethod
from torch.nn import Module


class Indicator(Module):
    '''
     @brief
     @note 每个epoch一个轮回，接收train阶段evaluate的decode输出，生成代表性指标，指导lr_reduce、auto_save、auto_stop等
    '''
    def __init__(self, args):
        Module.__init__(self)
        self._indicator_name = args.indicator_name
        self._indicator_source = args.indicator_source