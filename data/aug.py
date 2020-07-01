# coding=utf-8
from abc import ABCMeta, abstractmethod


class AugMethod(metaClass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        '''
         @brief return a aug func
        '''
        pass

    @abstractmethod
    def add_sub_aug_method(self, index, aug_method):
        pass

    @abstractmethod
    def reset_field(self):
        pass
    pass