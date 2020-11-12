# coding=utf-8

from torch.nn import Module


class Encode(Module):
    def __init__(self, args):
        self._encode_name = args.encode_name
        self._encode_source = args.encode_source
