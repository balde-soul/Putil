# coding=utf-8
import copy
from torch import optim
from abc import ABCMeta, abstractmethod


def common_optimization_arg(parser):
    parser.add_argument('--lr', type=float, action='store', default=0.001, \
        help='the basical learning rate')
    pass


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
            optimizer = optim.Adam(parameters, lr=self._args.lr, weight_decay=self._args.weight_decay)
            return optimizer
            pass
        else:
            raise NotImplementedError('optimization in framework {} is not implemented'.format(self._args.framework))
        pass
    pass


class DefaultOptimization(Optimization):
    def __init__(self, args):
        Optimization.__init__(self, args)
        self._args = args
        pass

    def __call__(self, parameters):
        return optim.SGD(parameters, self._args.lr)
    pass
#
#
#def DefaultOptimization(args):
#    temp_args = copy.deepcopy(args)
#    def generate_default_optimization():
#        return _DefaultOptimization(temp_args)
#    return generate_default_optimization


def DefaultOptimizationArg(parser):
    common_optimization_arg(parser)
    pass