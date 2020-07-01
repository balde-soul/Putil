# coding=utf-8
from abc import ABCMeta, abstractmethod


class AugField(metaClass=ABCMeta):
    def __init__(self):
        pass

    def pop(self, item):
        pass

    def push(self, item):
        pass
    pass

class AugMethod(metaClass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        '''
         @brief return a lambda func
        '''
        pass

    @abstractmethod
    def field(self):
        pass

    @abstractmethod
    def reset_field(self):
        pass

    @abstractmethod
    def erase_item(self):
        pass
    pass

class Aug(metaClass=ABCMeta):
    def __init__(self):
        pass

    def aug_field(self):
        '''
         @brief return a AugMethod
        '''
        pass
    pass