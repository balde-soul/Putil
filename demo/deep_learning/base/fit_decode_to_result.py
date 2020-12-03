# coding=utf-8
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('fit_decode_to_result').logger()
logger.setLevel(plog.DEBUG)


class DefaultFitDecodeToResult:
    def __init__(self):
        pass
    
    def __call__(self, datas):
        '''
         @brief generate the input for the backbone
        '''
        pass

def DefaultFitDecodeToResult(args):
    '''
     @param[in] args
    '''
    raise NotImplementedError('not implemented')


def DefaultFitDecodeToResultArg(parser):
    pass
