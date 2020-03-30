# coding=utf-8
from abc import ABC, abstractmethod
import Putil.base.logger as plog
from colorama import Fore


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
    pass

class AutoStop:
    @staticmethod
    def generate_args(parser):
        parser.add_argument('--auto_stop_patience', dest='AutoStopPatience', type=int, action='store', default=8, 
        help='when the time count to patience, it would stop')
        parser.add_argument('--auto_stop_mode', dest='AutoStopMode', type=str, action='store', default='max', 
        help='max or min, the change meaning the indicator is better, if the change do not match the mode, \
        we would cound the time')
        pass

    @staticmethod
    def get_patience_from_args(args):
        return args.AutoStopPatience
        pass

    @staticmethod
    def get_mode_from_args(args):
        return args.AutoStopMode
        pass

    @staticmethod
    def generate_AutoStop_from_args(args):
        params = dict()
        params['patience'] = args.AutoStopPatience
        params['mode'] = args.AutoStopMode
        return AutoStop(**params)
        pass

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
            return False
        pass
    pass