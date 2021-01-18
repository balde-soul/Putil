# coding=utf-8
import copy
import torch
from torch.nn import Module


def common_backend_arg(parser):
    parser.add_argument('--backend_arch', type=str, action='store', default='', \
        help='the arch for the backend')
    pass


class Backend:
    def __init__(self, args):
        pass
    pass


class _DefaultBackend(Backend, Module):
    def __init__(self, args):
        Backend.__init__(self, args)
        Module.__init__(self)
        self._b = torch.nn.Parameter(torch.Tensor([0.2]), requires_grad=True)
        self.register_parameter('b', self._b)
        pass

    def forward(self, x):
        return self._b * x
    pass


def DefaultBackend(args):
    temp_args = copy.deepcopy(args)
    def generate_default_backend():
        return _DefaultBackend(args)
    return generate_default_backend


def DefaultBackendArg(parser):
    common_backend_arg(parser)
    pass