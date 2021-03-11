# coding=utf-8
import Putil.data.aug as paug


def DefaultAug(args, property_type='', **kwargs):
    return paug.AugFuncNoOp


def DefaultAugArg(parser, property_type='', **kwargs):
    pass