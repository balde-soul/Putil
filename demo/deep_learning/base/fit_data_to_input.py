# coding=utf-8
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('fit_data_to_input').logger()
logger.setLevel(plog.DEBUG)


class DefaultFitDataToInput:
    def __init__(self):
        pass
    
    def __call__(self, datas):
        '''
         @brief generate the input for the backbone
        '''
        pass

def DefaultFitDataToInput(args):
    '''
     @param[in] args
    '''
    raise NotImplementedError('not implemented')


def DefaultFitDataToInputArg(parser):
    pass

