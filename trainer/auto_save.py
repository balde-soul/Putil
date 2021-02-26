# coding=utf-8

from abc import ABC, ABCMeta, abstractmethod
import numpy as np
import Putil.base.logger as plog
from colorama import Fore
from Putil.trainer.auto_save_args import generate_args

auto_save_logger = plog.PutilLogConfig("auto_save").logger()
auto_save_logger.setLevel(plog.DEBUG)
AutoSaveLogger = auto_save_logger.getChild("AutoSaveLogger")


class auto_save(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dice):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def save_or_not(self, indicator):
        '''
         @brief check the indicator and decide to save or not
         @note
         @ret
         True: save operation is supposed
         False: save operation is not supposed
        '''
        pass
    pass


class AutoSave(auto_save):
    @staticmethod
    def _direction_info(improve):
        return 'direction : {0}'.format('Up' if improve else 'Down')

    @staticmethod
    def _delta_info(delta):
        return 'delta : {0}'.format(delta)

    @staticmethod
    def generate_args(parser, property_type):
        generate_args(parser, property_type)
        pass

    @staticmethod
    def get_mode_from_args(args):
        return args.auto_save_mode

    @staticmethod
    def get_delta_from_args(args):
        return args.auto_save_delta

    @staticmethod
    def get_keep_save_range_from_args(args):
        return args.auto_save_keep_save_range

    @staticmethod
    def get_abandon_range_from_args(args):
        return args.auto_save_abandon_range

    @staticmethod
    def get_base_line_from_args(args):
        return args.auto_save_base_line

    @staticmethod
    def get_limit_line_from_args(args):
        return args.auto_save_limit_line

    @staticmethod
    def get_history_amount_from_args(args):
        return args.auto_save_history_amount

    @staticmethod
    def generate_AutoSave_from_args(args, property_type='', **kwargs):
        params = dict()
        params['mode'] = eval('args.{}auto_save_mode'.format(property_type))
        params['delta'] = eval('args.{}auto_save_delta'.format(property_type))
        params['keep_save_range'] = eval('args.{}auto_save_keep_save_range'.format(property_type))
        params['abandon_range'] = eval('args.{}auto_save_abandon_range'.format(property_type))
        params['base_line'] = eval('args.{}auto_save_base_line'.format(property_type))
        params['limit_line'] = eval('args.{}auto_save_limit_line'.format(property_type))
        params['history_amount'] = eval('args.{}auto_save_history_amount'.format(property_type))
        return AutoSave(**params)

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, mode='max', delta=0.001, keep_save_range=[], abandon_range=[], base_line=None, limit_line=None, history_amount=10):
        """
         @brief
         @note
         @param[in] mode
            represent the direction of the target, True means the larger indicator is better, False means the smaller indicator is better
         @param[in] delta: represent the threshold , while the value which the target indicator improve is larger than the delta, we want to save the short cut
         @param[in] keep_save_range(not implemented): while the step is inside keep_save_range, we want to save the short cut anyway
         @param[in] abandon_range(not implemented): while the step is inside abandon_range, we do not want to save the short cut anyway
         @param[in] base_line(not implemented): while the target indicator is worst than the base_line , we suggest not to save the short cut
         @param[in] limit_line(not implemented): while the target indicator is better than the limit_line , we suggest not to save the short cut
         @param[in] history_amount: the amount of the hitorical indicator which suggest to save the short cut we keep
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

        self._mode = mode 
        self._delta = delta
        self._history_amount = history_amount
        self._best_collection = []
        self._best = None

        self._direction = 1 if self._mode == 'max' else -1
        #   the function which receive the best indicator and the target indicator, return the flag which represent
        #   the target > best indicator (True) or target < best indicator (Flase)
        #   self._compare(best_indicator, target_indicaotr)
        #   use register_comparator to register the funtion
        self._comparator = None
        
        assert ((base_line is None) ^ (limit_line is None)) is False
        assert ((base_line is not None) ^ (limit_line is not None)) is False
        self._base_line = base_line
        self._limit_line = limit_line
        if base_line is not None:
            if base_line < limit_line == (self._direction == 1):
                pass
            else:
                raise ValueError('base_line vs. limit_line : {0} vs. {1} base on : {2}'.format(base_line, limit_line, self._direction_info(improve)))
            pass
        pass

    def register_comparator(self, comparator):
        self._comparator = comparator

    @property
    def comparator(self):
        return self._comparator

    def info(self):
        AutoSaveLogger.info(plog.info_color('mode: {0}'.format(self._mode)))
        AutoSaveLogger.info(plog.info_color('delta: {0}'.format(self._delta)))
        AutoSaveLogger.info(plog.info_color('keep_save_range: {0}'.format(self._keep_save_range)))
        AutoSaveLogger.info(plog.info_color('abandon_range: {0}'.format(self._abandon_range)))
        AutoSaveLogger.info(plog.info_color('base_line: {0}'.format(self._base_line)))
        AutoSaveLogger.info(plog.info_color('limit_line: {0}'.format(self._limit_line)))
        AutoSaveLogger.info(plog.info_color('history_amount: {0}'.format(self._history_amount)))
        pass

    def save_or_not(self, indicator):
        #AutoSaveLogger.info(Fore.GREEN + '-->SaveOrNot' + Fore.RESET)
        if self._best is None:
            self._best = indicator
            AutoSaveLogger.info(Fore.YELLOW + 'save at first val' + Fore.RESET)
            #AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
            return True
        else:
            if ((self._best - indicator) * self._direction) <= -self._delta:
                AutoSaveLogger.info(Fore.GREEN + 'improve from {0} to {1}, save weight and collection to collection'.format(self._best, indicator)+ Fore.RESET)
                self._best_collection.append(self._best)
                self._best = indicator
                #AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
                #if self._base_line is not None:
                #    if self._best > self._base_line:
                #        if self._best < self._limit_line
                return True
            else:
                AutoSaveLogger.info(Fore.GREEN + 'NOT SAVE' + Fore.RESET)
                #AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
                return False
            pass
        pass

    def state_dict(self):
        state_dict = {}
        state_dict['keep_save_range'] = self._keep_save_range
        state_dict['abandon_range'] = self._abandon_range
        state_dict['mode'] = self._mode
        state_dict['delta'] = self._delta
        state_dict['history_amount'] = self._history_amount
        state_dict['best_collection'] = self._best_collection
        state_dict['best'] = self._best
        state_dict['direction'] = self._direction
        state_dict['comparator'] = self._comparator
        state_dict['base_line'] = self._base_line
        state_dict['limit_line'] = self._limit_line
        return state_dict

    def load_state_dict(self, state_dict):
        self._keep_save_range = state_dict['keep_save_range']
        self._abandon_range = state_dict['abandon_range']
        self._mode = state_dict['mode']
        self._delta = state_dict['delta']
        self._history_amount = state_dict['history_amount']
        self._best_collection = state_dict['best_collection']
        self._best = state_dict['best']
        self._direction = state_dict['direction']
        self._comparator = state_dict['comparator']
        self._base_line = state_dict['base_line']
        self._limit_line = state_dict['limit_line']
        pass