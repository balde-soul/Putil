# coding=utf-8

from util.statistic_indicator import *


def statistic_indicator_factory(args):
    if args.statistic_indicatory_name == '':
        raise NotImplementedError('data_factory: {} is not implemented'.format(args.statistic_indicatory_name))
    return eval('{}(args)'.format(args.statistic_indicator_name))