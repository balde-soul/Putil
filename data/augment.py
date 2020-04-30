# coding=utf-8
from abc import ABCMeta, abstractmethod
import copy


class Augment(metaclass=ABCMeta):
    def __init__(self):
        self._config = None
        pass

    def set_config(self, config):
        self._pre_config(config)
        _old_config = copy.deepcopy(self._config)
        self._config = copy.deepcopy(config)
        self._post_config(_old_config)
        pass

    @abstractmethod
    def _pre_config(self, config):
        pass

    @abstractmethod
    def _post_config(self, config):
        pass
    pass