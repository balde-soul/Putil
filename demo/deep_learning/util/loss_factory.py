# coding=utf-8
from util.loss import *


def loss_factory(args):
    if args.loss_name == '':
        raise NotImplementedError('data_factory: {} is not implemented'.format(args.loss_name))
    return eval('{}(args)'.format(args.loss_name))