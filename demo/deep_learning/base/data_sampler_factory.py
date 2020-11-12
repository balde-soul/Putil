# conding=utf-8

from Putil.demo.deep_learning.base.data_loader import *


def data_sampler_factory(args):
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data_loader of framework: {} is not implemented'.format(args.framework))
    return eval('{}(args)'.format('{}_{}'.format(args.framework, DataSampler)))
