# coding=utf-8
import copy
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('fit_data_to_input').logger()
logger.setLevel(plog.DEBUG)

from Putil.demo.deep_learning.base import util


##@brief the FitToIndicator 提供一个方法，接收datas和output，
# datas代表着Dataset的输出，output代表着模型的输出，然后生成目标数据，传输给Loss，计算损失，该对象在Loss中进行调用
class FitToIndicator(metaclass=ABCMeta):
    ##@brief 
    # @param[in] args
    # @param[in] property_type
    # @param[in] kwargs
    def __init__(self, args, property_type='', **kwargs):
        pass

    def __call__(self, datas, output):
        return self._call_impl(datas, output)

    @abstractmethod
    def _call_impl(self, *kargs, **kwargs):
        pass

class _DefaultFitToIndicator(FitToIndicator):
    def __init__(self, args, property_type='', **kwargs):
        FitToIndicator.__init__(self, args, property_type, **kwargs)
        self._args = args
        pass
    
    def _call_impl(self, *kargs, **kwargs):
        '''
         @brief generate the input for the backbone
        '''
        return kargs[0][1], kargs[1]

def DefaultFitToIndicator(args, property_type='', **kwargs):
    '''
     @param[in] args
    '''
    temp_args = copy.deepcopy(args)
    def generate_default_fit_data_to_input():
        return _DefaultFitToIndicator(args, property_type, **kwargs)
    return generate_default_fit_data_to_input


def DefaultFitToIndicatorArg(parser, property_type='', **kwargs):
    pass

