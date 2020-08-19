# coding=utf-8
#In[]:
from abc import ABCMeta, abstractmethod
from enum import Enum


class IOConvertor(metaclass=ABCMeta):

    class IODirection(Enum):
        InputConvertion = 0
        OutputConvertion = 1
        Unknow = 2
        pass

    @abstractmethod
    def __call__(self, *args):
        '''
         @brief call this obj return the data for input
         @note
         @param[in] args
         the input data
        '''

    def __init__(self, io):
        self._io = io
        pass
    pass

ConvertToInput = IOConvertor

class IOConvertorNoOp(ConvertToInput):
    def __init__(self):
        ConvertToInput.__init__(self, IOConvertor.IODirection.Unknow)

    def __call__(self, *args):
        '''
         @brief call this obj return the data for input
         @note
         @param[in] args
         the input data
        '''
        return args
    pass

ConvertToInputNoOp = IOConvertorNoOp

#
#t = ConvertToInputNoOp()
#t(*t(1,))