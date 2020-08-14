# coding=utf-8

from abc import abstractmethod, ABCMeta


class DataTypeAdapter(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass
    pass


class DataTypeAdapterNoOp(DataTypeAdapter):
    def __init__(self):
        DataTypeAdapter.__init__(self)
        pass

    def __call__(sefl, *args):
        return args
    pass