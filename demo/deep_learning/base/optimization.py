# coding=utf-8
from abc import ABCMeta, abstractmethod


class Optimization(metaclass=ABCMeta):
    def __init__(self, args):
        self._args = args
        pass
        
    @abstractmethod
    def __call__(self, parameters):
        pass


class Adam(Optimization):
    def __init__(self, args):
        Optimization.__init__(self, args)
    
    def __call__(self, parameters):
        if self._args.framework == 'torch':
            import torch
            import horovod.torch as hvd
            # By default, Adasum doesn't need scaling up learning rate.
            lr_scaler = hvd.size() if not self._args.use_adasum else 1
            # Horovod: scale learning rate by lr_scaler. TODO:
            #optimizer = optim.SGD(model.parameters(), lr=self._args.lr_reduce_init_lr * lr_scaler, 
            #                      momentum=self._args.momentum, weight_decay=1e-4)
            optimizer = optim.Adam(model.parameters(), lr=self._args.lr_reduce_init_lr * lr_scaler, weight_decay=self._args.weight_decay)
            return optimizer
            pass
        else:
            raise NotImplementedError('optimization in framework {} is not implemented'.format(self._args.framework))
        pass
    pass


class DefaultOptimization(Optimization):
    def __init__(self, args):
        Optimization.__init__(self, args)
        pass
    pass


def DefaultOptimizationArg(parser):
    pass