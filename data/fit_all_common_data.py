# coding=utf-8
from abc import ABCMeta, abstractmethod

class FitAllCommonData(metaclass=ABCMeta):
    '''
     @brief fit the difference between the common_datas
     @note
     fit the difference between the common_datas, for example, the different order of the output between common_datas.
     use the ofs to get the target common_data , use the cofs to get the index of the target common_data
    '''
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, ofs, cofs):
        pass
    pass


class FitAllCommonDataNoOp(FitAllCommonData):
    def __init__(self):
        FitAllCommonData.__init__(self)
        pass

    def __call__(self, *args, ofs, cofs):
        return args
