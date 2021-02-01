# coding=utf-8
import copy
import torch
from torch.nn import Module


def common_backend_arg(parser, property_type='', **kwargs):
    parser.add_argument('--{}backend_arch'.format(property_type), type=str, action='store', default='', \
        help='the arch for the backend')
    pass


class Backend:
    def __init__(self, args, property_type='', **kwargs):
        pass
    pass


class _DefaultBackend(Backend, Module):
    def __init__(self, args, property_type='', **kwargs):
        Backend.__init__(self, args, property_type, **kwargs)
        Module.__init__(self)
        pass

    def forward(self, x):
        return torch.tanh(torch.mean(x, -1))
    pass


def DefaultBackend(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_default_backend():
        return _DefaultBackend(args, property_type, **kwargs)
    return generate_default_backend


def DefaultBackendArg(parser, property_type='', **kwags):
    common_backend_arg(parser, property_type, **kwargs)
    pass