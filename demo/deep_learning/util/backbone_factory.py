# coding=utf-8

from util.backbone import *


def backbone_factory(args):
    if args.backbone_name == '':
        raise NotImplementedError('backbone_name: {} is not implemented'.format(args.backbone_name))
    return eval('{}(args)'.format(args.backbone_name))