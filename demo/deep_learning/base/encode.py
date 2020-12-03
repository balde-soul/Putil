# coding=utf-8
import copy


class Encode:
    def __init__(self, args):
        self._encode_name = args.encode_name
        self._encode_source = args.encode_source
        pass
    pass


class _DefaultEncode(Encode):
    def __init__(self, args):
        Encode.__init__(self, args)
        Module.__init__(self)
        pass


def DefaultEncode(args):
    temp_args = copy.deepcopy(args)
    def generate_default_encode():
        return _DefaultEncode(temp_args)
    return generate_default_encode


def DefaultEncodeArg(parser):
    pass