# coding=utf-8
#In[]:
from abc import ABCMeta, abstractmethod


class ConvertToInput(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args):
        '''
         @brief call this obj return the data for input
         @note
         @param[in] args
         the input data
        '''

    def __init__(self):
        pass
    pass

class ConvertToInputNoOp(ConvertToInput):
    def __init__(self):
        ConvertToInput.__init__(self)

    def __call__(self, *args):
        '''
         @brief call this obj return the data for input
         @note
         @param[in] args
         the input data
        '''
        return args
    pass
#
#t = ConvertToInputNoOp()
#t(*t(1,))