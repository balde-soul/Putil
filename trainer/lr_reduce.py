# coding=utf-8

from abc import ABC, abstractmethod
import Putil.base.logger as plog
from colorama import Fore

lr_reduce_logger = plog.PutilLogConfig('auto_stop').logger()
lr_reduce_logger.setLevel(plog.DEBUG)
LrReduceLogger = lr_reduce_logger.getChild('AutoStop')
LrReduceLogger.setLevel(plog.DEBUG)

# while the indicator does not improve for patience epoch, this while return the reduce learn rate
# if feed None, would return the newest learn rate
# once the reduce worked, the reducer would coll down for lr_cool_down epoch, which not
class LrReduce:
    @staticmethod 
    def generate_args(parser):
        parser.add_argument('--lr_reduce_init_lr', dest='LrReduceInitLr', type=float, action='store', default=0.1, 
        help='the init lr, defautl: 0.1')
        parser.add_argument('--lr_reduce_lr_factor', dest='LrReduceLrFactor', type=float, action='store', default=0.1, 
        help='the reduce factor for lr reducion, default: 0.1')
        parser.add_argument('--lr_reduce_lr_epsilon', dest='LrReduceLrEpsilon', type=float, action='store', default=0.001, 
        help='the threshold which decide the reducion is triggered or not, default: 0.001')
        parser.add_argument('--lr_reduce_lr_patience', dest='LrReduceLrPatience', type=int, action='store', default=3, 
        help='while reducion is triggered in LrReduceLrPatience time, the lr whould be reduce, default: 3')
        parser.add_argument('--lr_reduce_lr_cool_down', dest='LrReduceLrCoolDown', type=int, action='store', default=2, 
        help='the epoch in which the reducion trigger would not be count, default: 2')
        parser.add_argument('--lr_reduce_lr_min', dest='LrReduceLrMin', type=float, action='store', default=0.0000001, 
        help='the minimum of the lr, default: 0.0000001')
        parser.add_argument('--lr_reduce_mode', dest='LrReduceMode', type=str, action='store', default='max', 
        help='the mode which we want the indicator to be, \
        if not or the transformation is not upper the LrReduceEpsilon, \
        the reducion would be triggered, defautl: max')
        pass

    @staticmethod
    def get_init_lr_from_args(args):
        return args.LrReduceInitLr
        pass

    @staticmethod
    def get_lr_factor_from_args(args):
        return args.LrReduceLrFactor
        pass

    @staticmethod
    def get_lr_epsilon_from_args(args):
        return args.LrReduceLrEpsilon
        pass

    @staticmethod
    def get_lr_patience_from_args(args):
        return args.LrReduceLrPatience
        pass

    @staticmethod
    def get_lr_cool_down_from_args(args):
        return args.LrReduceLrCoolDown
        pass

    @staticmethod
    def get_lr_min_from_args(args):
        return args.LrReduceLrMin
        pass

    @staticmethod
    def get_mode_from_args(args):
        return args.LrReduceMode
        pass

    @staticmethod
    def generate_LrReduce_from_args(args):
        params = dict()
        params['init_lr'] = args.LrReduceInitLr
        params['lr_factor'] = args.LrReduceLrFactor
        params['lr_epsilon'] = args.LrReduceLrEpsilon
        params['lr_patience'] = args.LrReduceLrPatience
        params['lr_cool_down'] = args.LrReduceLrCoolDown
        params['lr_min'] = args.LrReduceLrMin
        params['mode'] = args.LrReduceMode
        return LrReduce(**params)
        pass

    def __init__(self, init_lr, lr_factor, lr_epsilon, lr_patience, lr_cool_down, lr_min, mode='max'):
        LrReduceLogger.info(Fore.GREEN + '-->LrReduce.__init__' + Fore.RESET)
        self._lr_base = init_lr
        self._lr_now = init_lr
        self._lr_factor = lr_factor
        self._lr_epsilon = lr_epsilon
        self._lr_patience = lr_patience
        self._lr_cool_down = lr_cool_down
        self._lr_min = lr_min
        self._best = None
        self._count = 0
        self._cool_count = 0
        self._mode = mode
        self._direction = 1 if self._mode == 'max' else -1
        self._reduce = False
        LrReduceLogger.info(Fore.GREEN + 'LrReduce.__init__-->' + Fore.RESET)
        pass

    def info(self):
        LrReduceLogger.info(Fore.GREEN +
                            'lr_init: {0}'.format(self._lr_now)
                            + Fore.RESET
                            )
        LrReduceLogger.info(Fore.GREEN +
                             'lr_factor: {0}'.format(self._lr_factor)
                             + Fore.RESET
                             )
        LrReduceLogger.info(Fore.GREEN +
                             'lr_epsilon: {0}'.format(self._lr_epsilon)
                             + Fore.RESET)
        LrReduceLogger.info(Fore.GREEN +
                            'lr_patience: {0}'.format(self._lr_patience)
                            + Fore.RESET)
        LrReduceLogger.info(Fore.GREEN +
                            'lr_cool_down: {0}'.format(self._lr_cool_down)
                            + Fore.RESET)
        LrReduceLogger.info(Fore.GREEN +
                            'lr_min: {0}'.format(self._lr_min)
                            + Fore.RESET)
        LrReduceLogger.info(Fore.GREEN +
                            'direction: {0}({1})'.format(self._direction, self._mode)
                            + Fore.RESET)
        pass

    @property
    def LrBase(self):
        return self._lr_base

    @property
    def LrPatience(self):
        return self._lr_patience

    @property
    def LrFactor(self):
        return self._lr_factor

    @property
    def LrEpsilon(self):
        return self._lr_epsilon

    @property
    def LrCoolDown(self):
        return self._lr_cool_down

    @property
    def LrMin(self):
        return self._lr_min

    @property
    def Reduce(self):
        return self._reduce

    @property
    def LrNow(self):
        return self._lr_now

    def reduce_or_not(self, indicator):
        LrReduceLogger.info(Fore.GREEN + '-->reduce_or_not' + Fore.RESET)
        assert indicator is not None, LrReduceLogger.error(Fore.RED + 'indicator should not be None' + Fore.RESET)
        if self._cool_count != 0:
            LrReduceLogger.debug(Fore.RED + 'cool_count: {0}'.format(self._cool_count) + Fore.RESET)
            self._cool_count -= 1
            if (self._best - indicator) * self._direction < self._lr_epsilon:
                self._best = indicator
                pass
            else:
                pass
            plog.api_function_out_log(LrReduceLogger, 'reduce_or_not')
            return False
            pass
        else:
            if self._best is None:
                self._best = indicator 
                return False
                pass
            else:
                if (self._best - indicator) * self._direction < -self._lr_epsilon:
                    self._best = indicator
                    self._count = 0
                    pass
                else:
                    self._count += 1
                    pass
            if self._count >= self._lr_patience:
                self._cool_count = self._lr_cool_down
                LrReduceLogger.info(Fore.LIGHTGREEN_EX +
                                    'up to patience, reduce the learning rate from {0} to {1}'.format(
                                        self._lr_now, self._lr_factor * self._lr_now)
                                    + Fore.RESET
                                    )
                self._lr_now = self._lr_factor * self._lr_now
                self._count = 0
                LrReduceLogger.info(Fore.GREEN + 'reduce_or_not-->' + Fore.RESET)
                return True
            else:
                LrReduceLogger.info(Fore.GREEN + 'reduce_or_not-->' + Fore.RESET)
                return False
                pass
            pass
            LrReduceLogger.info(Fore.GREEN + 'reduce_or_not-->' + Fore.RESET)
        pass