# coding=utf-8
from abc import ABCMeta, abstractmethod

import Putil.base.logger as plog
logger = plog.PutilLogConfig('train_controler').logger()
logger.setLevel(plog.DEBUG)


class TrainControler(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, *kargs, **kwargs):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass 

    @property
    @abstractmethod
    def old_step(self):
        pass

    @property
    @abstractmethod
    def old_epoch(self):
        pass

    @property
    @abstractmethod
    def old_best_indicator(self):
        pass

    @property
    @abstractmethod
    def old_lr(self):
        pass
    pass


class DefaultTrainControler(TrainControler):
    def __init__(self):
        pass

    def update(self):
        pass