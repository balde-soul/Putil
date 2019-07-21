#coding=utf-8
from abc import ABC, abstractmethod


class auto_stop(ABC):
    '''
    this class is the virtual class for model training
    main function:
    
    '''
    def __init__(self, indicator_getter, patience  improve=True):
        self._indicator_getter = indicator_getter
        self._esp
        pass

    def Stop(self):

        pass

    @abstractmethod
    def _compare(self, Indicator):
        pass
    pass

    
