# coding=utf-8
from abc import ABCMeta, abstractmethod


class Comparator(metaclass=ABCMeta):
    def __init__(self):
        self._func = None
        pass

    def register_compare_func(self, func):
        self._func = func
        pass

    @abstractmethod
    def compare(self, a, b):
        return self._func(a, b)
        pass
    pass


class Comparator(Comparator):
    def __init__(self, mode='max', epsilon=None):
        pass
    pass