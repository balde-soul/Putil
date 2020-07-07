# coding=utf-8
from abc import ABC


class Function(ABC):
    def __init__(self):
        self._func = None
        pass

    def func(self):
        return self._func

    def calc(self, x):
        return self._func(x)

    def __call__(self, x):
        return self.calc(x)
    pass