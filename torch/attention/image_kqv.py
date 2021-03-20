# coding=utf-8
import torch

from Putil.torch.attention.KQV import *

class KQV2DPixel(KQV):
    def __init__(self, in_channels, mid_channels):
        key_func = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        query_func = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        value_func = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        KQV.__init__(self, [2, 3], key_func, query_func, value_func)
        pass
    pass


##@brief
# @note
class NonLocal(KQV2DPixel):
    def __init__(self, in_channels, mid_channels):
        KQV2DPixel.__init__(self, in_channels, mid_channels)
        self._rebuild = torch.nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        pass

    def forward(self, x):
        pixel_refine_with_attension = KQV2DPixel.forward(self, x)
        rebuild = self._rebuild(pixel_refine_with_attension)
        return torch.add(x, rebuild)
    pass


class KQV2DChannel(KQV):
    def __init__(self, in_channels, mid_channels):
        key_func = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(0, 0))
        query_func = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(0, 0))
        value_func = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, \
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(0, 0))
        KQV.__init__(self, [1], key_func, query_func, value_func)
        pass
    pass


class NonHeightNonWidthNoChannel(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, mid_channels):
        pass
    pass


class NonLocalNonChannel(NonLocal):
    def __init__(self, in_channels, mid_channels):
        NonLocal.__init__(self, in_channels, mid_channels)
        pass
