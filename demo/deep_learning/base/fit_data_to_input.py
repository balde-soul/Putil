# coding=utf-8
import copy
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('fit_data_to_input').logger()
logger.setLevel(plog.DEBUG)


class _DefaultFitDataToInput:
    def __init__(self, args, property_type='', **kwargs):
        pass
    
    def __call__(self, datas):
        '''
         @brief generate the input for the backbone
        '''
        return datas

def DefaultFitDataToInput(args, property_type='', **kwargs):
    '''
     @param[in] args
    '''
    temp_args = copy.deepcopy(args)
    def generate_default_fit_data_to_input():
        return _DefaultFitDataToInput(args, property_type, **kwargs)
    return generate_default_fit_data_to_input


def DefaultFitDataToInputArg(parser, property_type='', **kwargs):
    pass

