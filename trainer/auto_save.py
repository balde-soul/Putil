# coding=utf-8

from abc import ABC, ABCMeta
import numpy as np
import Putil.loger as plog
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
        parser.add_argument('--auto_saver_improve', dest='AutoSaverImprove', type=bool, action='store', default=True, help='the AutoSaverImprove, default: True')
        parser.add_argument('--auto_saver_delta', dest='AutoSaverDelta', type=float, action='store', default=0.001, help='the AutoSaverDelta, default: 0.001')
        parser.add_argument('--auto_saver_keep_save_range', dest='AutoSaverKeepSaveRange', type=list, action='store', default=[], help='the AutoSaverKeepSaveRange, default: []')
        parser.add_argument('--auto_saver_abandon_range', dest='AutoSaverAbandonRange', type=list, action='store', default=[], help='the AutoSaverAbandonRange, default: []')
        parser.add_argument('--auto_saver_base_line', dest='AutoSaverBaseLine', type=int, action='store', default=None, help='the AutoSaverBaseLine, default: None')
        parser.add_argument('--auto_saver_limit_line', dest='AutoSaverLimitLine', type=int, action='store', default=None, help='the AutoSaverLimitLine, default: None')
        parser.add_argument('--auto_saver_history_amount', dest='AutoSaverHistoryAmount', type=int, action='store', default=100, help='the AutoSaverHistoryAmount, default: 100')
        pass

    @staticmethod
    def get_improve_from_args(args):
        return args.AutoSaverImprove
        pass

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
        params['improve'] = args.AutoSaverImprove
        params['delta'] = args.AutoSaverDelta
        params['keep_save_range'] = args.AutoSaverKeepSaveRange
        params['abandon_range'] = args.AutoSaverAbandonRange
        params['base_line'] = args.AutoSaverBaseLine
        params['limit_line'] = args.AutoSaverLimitLine
        params['history_amount'] = args.AutoSaverHistoryAmount
        return AutoSave(**params)
        pass

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
        self._best_collection = []
        self._best = None

        self._direction = 1 if improve is True else -1
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
        AutoSaveLogger.info(self._direction_info(self._improve))
        AutoSaveLogger.info(self._delta_info(self._delta))

    def save_or_not(self, indicator):
        AutoSaveLogger.info(Fore.GREEN + '-->SaveOrNot' + Fore.RESET)
        if self._best is None:
            self._best = indicator
            AutoSaveLogger.info('save at first val')
            return True
            pass
        else:
            if (self._best - indicator) * self._direction < -self._delta:
                AutoSaveLogger.info(Fore.GREEN + 'improve from {0} to {1}, save weight and collection to collection'.format(self._best * self._direction, indicator)+ Fore.RESET)
                self._best_collection.append(self._best)
                self._best = indicator
                AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
                return True
            else:
                AutoSaveLogger.info(Fore.GREEN + 'SaveOrNot-->' + Fore.RESET)
                return False
            pass
        pass