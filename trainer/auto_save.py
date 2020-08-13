# coding=utf-8

from abc import ABC, ABCMeta
import numpy as np
import Putil.base.logger as plog
from colorama import Fore

auto_save_logger = plog.PutilLogConfig("auto_savE").logger()
auto_save_logger.setLevel(plog.DEBUG)
AutoSaveLogger = auto_save_logger.getChild("AutoSaveLogger")


class auto_save(metaclass=ABCMeta):
    def __init__(self):
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
    def generate_args(parser):
        parser.add_argument('--auto_save_mode', dest='AutoSaveMode', type=str, action='store', default='max', help='the AutoSaverImprove, default: True')
        parser.add_argument('--auto_save_delta', dest='AutoSaveDelta', type=float, action='store', default=0.001, help='the AutoSaverDelta, default: 0.001')
        parser.add_argument('--auto_save_keep_save_range', dest='AutoSaveKeepSaveRange', type=list, action='store', default=[], help='the AutoSaverKeepSaveRange, default: []')
        parser.add_argument('--auto_save_abandon_range', dest='AutoSaveAbandonRange', type=list, action='store', default=[], help='the AutoSaverAbandonRange, default: []')
        parser.add_argument('--auto_save_base_line', dest='AutoSaveBaseLine', type=int, action='store', default=None, help='the AutoSaverBaseLine, default: None')
        parser.add_argument('--auto_save_limit_line', dest='AutoSaveLimitLine', type=int, action='store', default=None, help='the AutoSaverLimitLine, default: None')
        parser.add_argument('--auto_save_history_amount', dest='AutoSaveHistoryAmount', type=int, action='store', default=100, help='the AutoSaverHistoryAmount, default: 100')
        pass

    @staticmethod
    def get_improve_from_args(args):
        return args.AutoSaverImprove

    @staticmethod
    def get_delta_from_args(args):
        return args.AutoSaverDelta
        pass

    @staticmethod
    def get_keep_save_range_from_args(args):
        return args.AutoSaverKeepSAverRange
        pass

    @staticmethod
    def get_abandon_range_from_args(args):
        return args.AutoSaverAbandonRange
        pass

    @staticmethod
    def get_base_line_from_args(args):
        return args.AutoSaverBaseLine
        pass

    @staticmethod
    def get_limit_line_from_args(args):
        return args.AutoSaverLimitLine
        pass

    @staticmethod
    def get_history_amount_from_args(args):
        return args.AutoSaverHistoryAmount
        pass

    @staticmethod
    def generate_AutoSave_from_args(args):
        params = dict()
        params['mode'] = args.AutoSaveMode
        params['delta'] = args.AutoSaveDelta
        params['keep_save_range'] = args.AutoSaveKeepSaveRange
        params['abandon_range'] = args.AutoSaveAbandonRange
        params['base_line'] = args.AutoSaveBaseLine
        params['limit_line'] = args.AutoSaveLimitLine
        params['history_amount'] = args.AutoSaveHistoryAmount
        return AutoSave(**params)
        pass

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, mode='max', delta=0.001, keep_save_range=[], abandon_range=[], base_line=None, limit_line=None, history_amount=10):
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
        self._base_line = base_line
        self._limit_line = limit_line
        if base_line is not None:
            if base_line < limit_line == improve:
                self._base_line = base_line
                self._limit_line = limit_line
            else:
                raise ValueError('base_line vs. limit_line : {0} vs. {1} base on : {2}'.format(base_line, limit_line, self._direction_info(improve)))
                pass
            pass

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
        AutoSaveLogger.info(Fore.GREEN + '-->SaveOrNot' + Fore.RESET)
        if self._best is None:
            self._best = indicator
            AutoSaveLogger.info(Fore.YELLOW + 'save at first val' + Fore.RESET)
            return True
            pass
        else:
            if ((self._best - indicator) * self._direction) <= -self._delta:
                AutoSaveLogger.info(Fore.GREEN + 'improve from {0} to {1}, save weight and collection to collection'.format(self._best, indicator)+ Fore.RESET)
                self._best_collection.append(self._best)
                self._best = indicator
                AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
                return True
            else:
                AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
                return False
            pass
        pass