# coding=utf-8

from abc import ABC
import numpy as np
import Putil.loger as plog

ROOT_LOGGER = plog.PutilLogConfig("auto_savE").logger()
ROOT_LOGGER.setLevel(plog.DEBUG)
AUTO_SAVE_LOGGER = ROOT_LOGGER.getChild("AutoSaveLogger")


class AutoSave(ABC):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, improve=True, delta=0.001, keep_save_range=[], abandon_range=[], base_line=None, limit_line=None, history_amount=10):
        """
        param:
            improve: represent the direction of the target, True means the larger indicator is better, False means the smaller indicator is better
            delta: represent the threshold , while the value which the target indicator improve is larger than the delta, we want to save the short cut
            keep_save_range: while the step is inside keep_save_range, we want to save the short cut anyway
            abandon_range: while the step is inside abandon_range, we do not want to save the short cut anyway
            base_line: while the target indicator is worst than the base_line , we suggest not to save the short cut
            limit_line: while the target indicator is better than the limit_line , we suggest not to save the short cut
            history_amount: the amount of the hitorical indicator which suggest to save the short cut we keep
        according to the information : the importance of the parameter: keep_save_range==abandon_range>base_line==limit_line
        """
        ksr = sorted(keep_save_range)
        a_r = sorted(abandon_range)
        same = [i for i in ksr if i in a_r]
        #   Do not use `len(SEQUENCE)` to determine if a sequence is empty
        if not same:
            self._keep_save_range = keep_save_range
            self._abandon_range = abandon_range
        else:
            raise ValueError('keep_save_range vs. abandon_range : {0} vs. {1}'.format(keep_save_range, abandon_range))
            pass
        
        assert ((base_line is None) ^ (limit_line is None)) is False
        assert ((base_line is not None) ^ (limit_line is not None)) is False
        if base_line is not None:
            if base_line < limit_line == improve:
                self._base_line = base_line
                self._limit_line = limit_line
            else:
                raise ValueError('base_line vs. limit_line : {0} vs. {1} base on : {2}'.format(base_line, limit_line, self._direction_info(improve)))
                pass
            pass

        self._improve = improve
        self._delta = delta
        self._history_amount = history_amount
        self._best_history = []
        self._best = None
        #   the function which receive the best indicator and the target indicator, return the flag which represent
        #   the target > best indicator (True) or target < best indicator (Flase)
        #   self._compare(best_indicator, target_indicaotr)
        #   use register_comparator to register the funtion
        self._comparator = None

    @staticmethod
    def _direction_info(improve):
        return 'direction : {0}'.format('Up' if improve else 'Down')

    @staticmethod
    def _delta_info(delta):
        return 'delta : {0}'.format(delta)

    def register_comparator(self, comparator):
        self._comparator = comparator

    @property
    def comparator(self):
        return self._comparator

    def info(self):
        AUTO_SAVE_LOGGER.info(self._direction_info(self._improve))
        AUTO_SAVE_LOGGER.info(self._delta_info(self._delta))

    def save(self, indicator):
        pass
