# coding=utf-8
from ..decode import Decode
from torch.nn import Module


class CenterNetDecode(Decode, Module):
    def __init__(self, args):
        Decode.__init__(self, args)
        Module.__init__(self)
        self._threshold = args.center_net_decode_threshold
        pass

    def forward(self, x):
        pass
    pass