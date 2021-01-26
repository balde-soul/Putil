# coding=utf-8
from abc import abstractmethod, ABCMeta
from torch.nn import Module
import torch
import copy


##@brief common_loss_arg generate the common arg parser for the loss func
# @param[in] parser the argparser
def common_loss_arg(parser):
    pass


class Loss(metaclass=ABCMeta):
    '''
     @brief
     @note 传入输入数据与模型输出，计算losses
    '''
    def __init__(self, args, property_type='', **kwargs):
        self._loss_name = args.loss_name
        self._loss_source = args.loss_source
        self._fit_to_loss_input = kwargs.get('fit_to_loss_input', None)

    @abstractmethod
    def total_loss_name(self):
        '''
         @brief return the string which would be used to calculate the grad
        '''
        pass

    def __call__(self, datas, output, **kwargs):
        kargs = self._fit_to_loss_input(datas, output) if self._fit_to_loss_input is not None else (datas, output)
        return self._call_impl(*kargs, **kwargs)

    @abstractmethod
    def _call_impl(self, *kargs, **kwargs):
        pass

    @property
    def fit_to_loss(self):
        return self._fit_to_loss
    pass


class _DefaultLoss(Loss, Module):
    def __init__(self, args, property_type='', **kwargs):
        Loss.__init__(self, args, property_type, **kwargs)
        Module.__init__(self)
        pass

    def _call_impl(self, *args, **kwargs):
        label = args[0]
        output = args[1]
        return {'total_loss': torch.mean((torch.squeeze(label) - torch.squeeze(output)) ** 2)}

    @property
    def total_loss_name(self):
        return 'total_loss'
    pass


def DefaultLoss(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_default_loss():
        return _DefaultLoss(args, property_type, **kwargs)
    return generate_default_loss


def DefaultLossArg(parser):
    pass