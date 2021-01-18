# coding=utf-8
from abc import abstractmethod, ABCMeta
from torch.nn import Module
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
    def __init__(self, args):
        self._loss_name = args.loss_name
        self._loss_source = args.loss_source

    @abstractmethod
    def total_loss_name(self):
        '''
         @brief return the string which would be used to calculate the grad
        '''
        pass
    pass


class _DefaultLoss(Loss, Module):
    def __init__(self, args):
        Loss.__init__(self, args)
        pass

    def __call__(self, x):
        return x

    def total_loss_name(self):
        return 'loss'
    pass


def DefaultLoss(args):
    temp_args = copy.deepcopy(args)
    def generate_default_loss():
        return _DefaultLoss(args)
    return generate_default_loss


def DefaultLossArg(parser):
    pass