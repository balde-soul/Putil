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
        '''
         @brief
         @note
         @param[in] io
         IOConvertor.IODirection, represent the direction of the data
         InputConvertion: change the data from GeneralData(which could be generated by DataCommon) to NetIn
         OutputConvertion: change the data from NetOut to GeneralData(which could be used by DataCommon)
         Unknow: do nothing
        '''
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