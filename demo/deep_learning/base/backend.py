# coding=utf-8
import copy
from torch.nn import Module


class Backend:
    def __init__(self, args):
        pass
    pass


class _DefaultBackend(Backend, Module):
    def __init__(self, args):
        Backend.__init__(self, args)
        Module.__init__(self)
        pass

    def forward(self, x):
        return x
    pass


def DefaultBackend(args):
    temp_args = copy.deepcopy(args)
    def generate_default_backend():
        return _DefaultBackend(args)
    return generate_default_backend


def DefaultBackendArg(parser):
    pass