# coding=utf-8
from abc import ABC


class Function(ABC):
    def __init__(self):
        self._func = None
        pass

    def func(self):
        return self._func
        pass

    def calc(self, x):
        return self._func(x)
        pass
    pass