# coding=utf-8
from torch.nn import Module


class Loss(Module):
    '''
     @brief
     @note 传入输入数据与模型输出，计算losses
    '''
    def __init__(self, args):
        self._loss_name = args.loss_name
        self._loss_source = args.loss_source

    def output_reflect(self):
