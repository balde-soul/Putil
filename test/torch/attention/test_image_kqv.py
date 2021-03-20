# coding=utf-8

import torch

from Putil.torch.attention import image_kqv

in_channels = 4
mid_channels = 2

_input = torch.rand([1, in_channels, 10, 10], dtype=torch.float32)

attention_pixel = image_kqv.NonLocal(in_channels=in_channels, mid_channels=mid_channels)
_out = attention_pixel(_input)
_attention = attention_pixel.attention
print('done')