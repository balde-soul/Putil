# coding=utf-8
from Putil.demo.deep_learning.base.optimization import *


def optimization_factory(args):
    '''
     @note generate
    '''
    if args.optimization_name == '':
        pass
    else:
        raise NotImplementedError('auto save: {} is not implemented'.format(args.optimization_name))
    return eval('{}(args)'.format(args.optimization_name))
