# coding=utf-8
from abc import ABCMeta, abstractmethod


class aug_node(metaClass=ABCMeta):
    '''
    '''
    def __init__(self, aug_method):
        self._children = list()
        self._aug_method = aug_method
        pass

    def __len__(self):
        pass
    
    def __item__(self, index):
        pass

    def child_amount(self):
        return len(self._children)

    def children(self):
        return self._children

    def child(self, index):
        return self._children[index]

    def add_child(self, aug_method):
        self._children.append(aug_method)

    @staticmethod
    def empty_node():
        return aug_node()
    pass


class AugMethod(metaClass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        '''
         @brief return the size of the func
        '''
        pass

    @abstractmethod
    def __getitem__(self, index):
        '''
         @brief return a aug func
        '''
        pass
    pass