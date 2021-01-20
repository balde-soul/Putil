# coding=utf-8
import copy
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('fit_decode_to_result').logger()
logger.setLevel(plog.DEBUG)


class DefaultFitDecodeToResult:
    def __init__(self, args, property_type='', **kwargs):
        pass
    
    def __call__(self, datas):
        '''
         @brief generate the input for the backbone
        '''
        return datas

def DefaultFitDecodeToResult(args, property_type='', **kwargs):
    '''
     @param[in] args
    '''
    temp_args = copy.deepcopy(args)
    def generate_default_fit_decode_to_result():
        return DefaultFitDecodeToResult(temp_args, property_type, **kwargs)
    return generate_default_fit_decode_to_result


def DefaultFitDecodeToResultArg(parser, property_type='', **kwargs):
    pass
