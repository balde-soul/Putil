# coding=utf-8
import Putil.base.error as error


class TrainerError(error.PutilError):
    def __init__(self):
        super().__init__()
        pass
    pass


from abc import ABC, abstractmethod


class A(ABC):
    def __init__(self):
        print('a')
        pass

    def f(self):
        print('f')
        pass

    @abstractmethod
    def fb(self):
        pass
    pass


class B(A):
    def __init__(self):
        super().__init__()
        print('b')
        pass

    def fb(self):
        print('b')
        pass
    pass


def t(a):
    a.fb()
    pass


if __name__ == '__main__':
    t(A())
