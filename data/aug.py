# coding=utf-8
from abc import ABCMeta, abstractmethod

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

def no_op(*args):
    return args

class aug_node(metaClass=ABCMeta):
    '''
    '''
    def __init__(self, aug_func):
        self._children = list()
        self._aug_func = aug_func
        self._parent = None
        self._freezed = False

        self._func = list()
        pass

    def __len__(self):
        if self._freezed:
            child_len = 1
            for child in self._children:
                child_len += len(child)
                pass
            pass
        else:
            child_len = len(self._func)
            pass
        return child_len
    
    def __item__(self, index):
        pass

    def child_amount(self):
        return len(self._children)

    def children(self):
        return self._children

    def child(self, index):
        return self._children[index]
    
    def parent(self):
        return self._parent
    
    def set_parent(self, parent):
        raise RuntimeError('node has been freezed') if self._freezed else None
        self._parent = parent
        pass

    def add_child(self, aug_method):
        raise RuntimeError('node has been freezed') if self._freezed else None 
        self._children.append(aug_node(aug_method))
        self._children[-1].set_parent(self)
        pass

    def freeze_node(self):
        self._freezed = True
        for child in self._children:
            child.freeze_node()
            pass
        pass

    def _generate_the_func_queue(self):
        pass

    @staticmethod
    def empty_node():
        return aug_node(no_op)
    pass
