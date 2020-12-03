# coding=utf-8
import Putil.data.aug as paug


def DefaultAug(args):
    return paug.AugFuncNoOp


def DefaultAugArg(parser):
    pass