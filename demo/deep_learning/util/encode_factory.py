# coding=utf-8
from util.encode import *


def encode_factory(args):
    if args.encode_name == '':
        raise NotImplementedError('data_factory: {} is not implemented'.format(args.encode_name))
    return eval('{}(args)'.format(args.encode_name))