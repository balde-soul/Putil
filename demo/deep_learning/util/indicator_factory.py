# conding=utf-8

from util.indicator import *


def indicator_factory(args):
    if args.indicator_name == '':
        raise NotImplementedError('data_factory: {} is not implemented'.format(args.indicator_name))
    return eval('{}(args)'.format(args.indicator_name))