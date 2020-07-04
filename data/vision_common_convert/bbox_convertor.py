# coding=utf-8
import Putil.data.convert_to_input as convert_to_input

class BBoxConvertToInputMethod(convert_to_input.ConvertToInput):
    def __init__(self, config):
        '''
        '''
        convert_to_input.ConvertToInput.__init__(self)
        pass

    def __call__(self, *args):
        return args
    pass