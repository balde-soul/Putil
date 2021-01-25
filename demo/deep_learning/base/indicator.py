'''
 @note
 本文件提供指标计算方法，该指标应用于train与evaluate中的量化，通过该指标判断train与evaluate的基本情况
 以及进行evaluate是否提升的判断
'''
# coding=utf-8
from abc import abstractmethod
from torch.nn import Module
import torch
import copy


class Indicator:
    '''
     @brief
     @note 每个epoch一个轮回，接收train阶段evaluate的decode输出，生成代表性指标，指导lr_reduce、auto_save、auto_stop等
    '''
    def __init__(self, args, property_type='', **kwargs):
        self._indicator_name = args.indicator_name
        self._indicator_source = args.indicator_source
        self._fit_to_indicator = kwargs.get('fit_to_indicator', None)
        pass

    def __call__(self, datas, output):
        '''
         @brief
         @note
         @param[in] input contain the ground truth and the prediction
         @ret return a dict, {str: value}, the value can be reduce
        '''
        kargs = self._fit_to_indicator(datas, output) if self._fit_to_indicator(datas, output) else (datas, output)
        return self._call_impl(*kargs)

    @abstractmethod
    def _call_impl(self, *args, **kwargs):
        pass

    @property
    def fit_to_indicator(self):
        return self._fit_to_indicator
    pass


class _DefaultIndicator(Indicator, Module):
    def __init__(self, args, property_type='', **kwargs):
        Indicator.__init__(self, args, property_type, **kwargs)
        Module.__init__(self)
        pass
    
    def _call_impl(self, *args, **kwargs):
        label = args[0]
        output = args[1]
        return {'dist': torch.mean((torch.squeeze(label) - torch.squeeze(output)) ** 2)}
    pass


def DefaultIndicator(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_default_indicator():
        return _DefaultIndicator(temp_args, properyt_type, **kwargs)
    return generate_default_indicator


def DefaultIndicatorArg(parser, property_type='', **kwargs):
    pass