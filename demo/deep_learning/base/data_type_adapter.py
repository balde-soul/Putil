# coding=utf-8


def _DefaultDataTypeAdapter(*args):
    return args


def DefaultDataTypeAdapter(args, property_type='', **kwargs):
    def generate_default_data_type_adapter():
        return _DefaultDataTypeAdapter
    return generate_default_data_type_adapter


def DefaultDataTypeAdapterArg(parser, property_type='', **kwargs):
    pass
