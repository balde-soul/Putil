# coding=utf-8
import copy


class Encode:
    def __init__(self, args, property_type='', **kwargs):
        pass
    pass


class _DefaultEncode(Encode):
    def __init__(self, args, property_type='', **kwargs):
        Encode.__init__(self, args, property_type, **kwargs)
        pass

    def __call__(self, *args):
        return args


def DefaultEncode(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_default_encode():
        return _DefaultEncode(temp_args)
    return generate_default_encode


def DefaultEncodeArg(parser, property_type='', **kwargs):
    pass