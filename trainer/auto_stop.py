# coding=utf-8
from abc import ABC, abstractmethod
import Putil.base.logger as plog
from colorama import Fore
from Putil.trainer.auto_stop_args import generate_args


auto_stop_logger = plog.PutilLogConfig('auto_stop').logger()
auto_stop_logger.setLevel(plog.DEBUG)
AutoStopLogger = auto_stop_logger.getChild('AutoStop')
AutoStopLogger.setLevel(plog.DEBUG)

class auto_stop(ABC):
    '''
    this class is the virtual class for model training
    main function:
    
    '''
    def __init__(self):
        self._compare_func = None
        self._decision_func = None
        self._process_func = None
        pass
    
    def _set_compare_func(self, func):
        self._compare_func = func
        pass
    
    def Stop(self, indicator):
        # compare
        # process
        # decision
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    pass

class AutoStop:
    @staticmethod
    def generate_args(parser, property_type):
        generate_args(parser, property_type)
        pass

    @staticmethod
    def get_patience_from_args(args):
        return args.auto_stop_patience

    @staticmethod
    def get_mode_from_args(args):
        return args.auto_stop_mode

    @staticmethod
    def generate_AutoStop_from_args(args, property_type='', **kwargs):
        params = dict()
        params['patience'] = eval('args.{}auto_stop_patience'.format(property_type))
        params['mode'] = eval('args.{}auto_stop_mode'.format(property_type))
        return AutoStop(**params)

    def __init__(self, patience, mode='max'):
        self._patience = patience
        self._best = None
        self._count = 0
        self._mode = mode
        self._direction = 1 if self._mode == 'max' else -1
        AutoStopLogger.info(Fore.GREEN +'patience: {0}'.format(self._patience) + Fore.RESET)
        pass

    def info(self):
        AutoStopLogger.info(plog.info_color('patience: {0}'.format(self._patience)))
        AutoStopLogger.info(plog.info_color('mode: {0}'.format(self._mode)))
        pass

    @property
    def Patience(self):
        return self._patience

    def stop_or_not(self, value):
        if self._best is None:
            AutoStopLogger.debug(Fore.GREEN + 'best is None now, and base indicator: {0}'.format(value) + Fore.RESET)
            self._best = value * self._direction
            pass
        else:
            if self._best < value * self._direction:
                self._count = 0
                self._best = value * self._direction
                pass
            else:
                self._count += 1
                pass
            pass
        if self._count >= self._patience:
            AutoStopLogger.info(Fore.GREEN + 'not improve for {0}, auto stop the training'.format(self._patience) + Fore.RESET)
            return True
        else:
            AutoStopLogger.info(Fore.GREEN + 'NOT STOP' + Fore.RESET)
            return False
        pass

    def state_dict(self):
        state_dict = {}
        state_dict['best'] = self._best
        state_dict['direction'] = self._direction
        state_dict['count'] = self._count
        state_dict['patience'] = self._patience
        state_dict['mode'] = self._mode
        return state_dict

    def load_state_dict(self, state_dict):
        self._best = state_dict['best']
        self._direction = state_dict['direction']
        self._count = state_dict['count']
        self._patience = state_dict['patience']
        self._mode = state_dict['mode']
        pass
    pass