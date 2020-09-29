# coding=utf-8

from util.decode import *


def decode_factory(args):
    if args.decode_name == '':
        raise NotImplementedError('data_factory: {} is not implemented'.format(args.decode_name))
    return eval('{}(args)'.format(args.decode_name))