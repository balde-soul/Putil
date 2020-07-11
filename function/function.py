# coding=utf-8
from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self):
        self._func = None
        pass

    @abstractmethod
    def __call__(self, x):
        pass
    pass