# coding=utf-8

import torch

from Putil.torch.attention import KQV

in_channels = 4
mid_channels = 2

_input = torch.rand([1, in_channels, 10, 10], dtype=torch.float32)

attention_pixel = KQV.KQV2DPixel(in_channels=in_channels, mid_channels=mid_channels)
_out = attention_pixel(_input)
_attention = attention_pixel.attention
print('done')

attention_nonlocal = KQV.NonLocal(in_channels=in_channels, mid_channels=mid_channels)