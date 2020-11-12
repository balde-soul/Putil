# coding=utf-8

from torch.nn import Module


class Decode(Module):
    def __init__(self, args):
        self._decode_name = args.decode_name
        self._decode_source = args.decode_source

