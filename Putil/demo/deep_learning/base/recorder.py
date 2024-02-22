# coding=utf-8
import copy
from abc import ABCMeta, abstractmethod

import Putil.base.logger as plog
logger = plog.PutilLogConfig('recorder').logger()
logger.setLevel(plog.DEBUG)


class Recorder(metaclass=ABCMeta):
    def __init__(self):
        self._epoch = 0
        self._step = 0
        pass

    def epoch_getter(self):
        return self._epoch
    def epoch_setter(self, epoch):
        self._epoch = epoch
    epoch = property(epoch_getter, epoch_setter)

    def step_getter(self):
        return self._step
    def step_setter(self, step):
        self._step = step
    step = property(step_getter, step_setter)

    def state_dict(self):
        return {
            'epoch': self._epoch,
            'step': self._step
        }
        pass

    def load_state_dict(self, state_dict):
        self._epoch = state_dict['epoch']
        self._step = state_dict['step']
        pass
    pass


class _DefaultRecorder(Recorder):
    def __init__(self, args):
        Recorder.__init__(self)
        pass
    pass


def DefaultRecorder(args):
    temp_arg = copy.deepcopy(args)
    def generate_default_recorder():
        return _DefaultRecorder(temp_arg)
    return generate_default_recorder


def DefaultRecorderArg(parser):
    pass