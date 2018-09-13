# coding=utf-8
import Putil.loger as plog
import six
import abc
from colorama import Fore
import sys

root_logger = plog.PutilLogConfig('tf/param/early_stop').logger()
root_logger.setLevel(plog.DEBUG)
ValAccLogger = root_logger.getChild('ValAcc')
ValAccLogger.setLevel(plog.DEBUG)
EarlyStopClassLogger = root_logger.getChild('EarlyStopClass')
EarlyStopClassLogger.setLevel(plog.DEBUG)


"""
                common api
                     |
                abs EarlyStop
                     |
                menage by param
                     |
     ------------------------------------
    |                |           |       |
    |                |           |       |
early_stop_1    early_stop_2    ....    ....
"""
"""
EarlyStop usage:
    Generator EarlyStop in Model class ,and use it while in training
    
"""

@six.add_metaclass(abc.ABCMeta)
class EarlyStop:
    @abc.abstractmethod
    def CheckEarlyStop(self):
        """
        an abstract method to check the object is Ok or not
        :return: bool
        """
        return False
        pass

    @abc.abstractmethod
    def Stop(self):
        """
        an abstract method which return the bool which represent the flag of stop or not
        :return: bool : True: stop False: stop
        """
        return False
        pass

    @abc.abstractmethod
    def SetIndicatorGet(self, getter):
        """
        an abstract method which provide the method to set the indicator getter
        :param getter:
        :return:
        """
        return self
        pass

    @abc.abstractmethod
    def SetDecider(self, decider):
        """
        an bastract method which provide way to set the decider
        :param decider: function
        :return:self
        """
        pass
    pass


"""
    this class is use for generator a new EarlyStop object
"""


class EarlyStopClass(EarlyStop):
    def __init__(self):
        self._decider = None
        self._indicator_getter = None
        pass

    def Stop(self):
        return self._decider(self._indicator_getter())

    def SetDecider(self, decider):
        self._decider = decider
        return self

    def SetIndicatorGet(self, getter):
        self._indicator_getter = getter
        return self
    pass

    def CheckEarlyStop(self):
        if self._indicator_getter is None:
            EarlyStopClassLogger.fatal(Fore.LIGHTRED_EX + 'you should set the indicator getter by using the method'
                                                          'SetIndicatorGet' + Fore.RESET)
            sys.exit()
        else:
            pass
        if self._decider is None:
            EarlyStopClassLogger.fatal(Fore.LIGHTRED_EX + 'you should set the decider by using the method '
                                                          'SetDecider' + Fore.RESET)
            sys.exit()
        else:
            pass
        return self
        pass


"""
usage:this class provide a more sample way to apply EarlyStop , is frequently-used
class model:
    def __init__(self):
        self._val_acc_stop = ValAccStop().UseDefaultDecider(threshold=0.0001, interval=10, max=True).SetIndicatorGet(self.Output['acc'])

    @property
    def Output(self):
        return {'acc': None, "loss": None}
        pass
"""

# tested


class ValAccStop(EarlyStopClass):
    """provide the default method"""
    def __init__(self):
        EarlyStopClass.__init__(self)
        self._record = None
        self._target_count = 0
        self._threshold = None
        self._interval = None
        self._regular = None
        self._max = None
        self._min = None
        pass

    def UseDefaultDecider(self, **options):
        """
        set the Decider parameter
        this class provide a frequent-used Decider(while no improved up to threshold for n epoch val
            Stop while return a True, in this way, 'improve' could be indicator rise or indicator descend,
        :param options:
            threshold: the threshold which decide improve woork or not
            interval: patience for interval val epoch
            max: treats indicator rise as improved
            min: treats indicator descend as improved
            *max and min can not be same at the same time*
        :return:
        """
        self._threshold = options.pop('threshold', self._threshold)
        self._interval = options.pop('interval', self._interval)
        self._max = options.pop('max', self._max)
        self._min = options.pop('min', self._min)
        if self._max == self._min:
            ValAccLogger.fatal(Fore.LIGHTGREEN_EX +
                               'max and min should not be the same '
                               'you should set max or min at once'
                               + Fore.RESET
                               )
            raise ValueError()
        self._regular = True if self._max is True else self._regular
        self._regular = False if self._min is True else self._regular
        self._decider = self._default_decider
        return self
        pass

    def _default_decider(self, val_acc):
        old = self._record
        if self._record is None:
            self._record = val_acc
        else:
            if (val_acc - self._record >= self._threshold) == self._regular:
                self._target_count = 0
                self._record = val_acc
                pass
            else:
                self._target_count += 1
                self._record = val_acc
                pass
            pass
        if self._target_count == self._interval:
            ValAccLogger.info(Fore.LIGHTGREEN_EX +
                              'early stop: '
                              '\nstop target: {0}'
                              '\nhistory target: {1}'
                              '\nregular: {2}'
                              '\ninterval: {3}'.format(
                                  val_acc,
                                  old,
                                  'max' if self._regular is True else 'min',
                                  self._interval))
            return True
        else:
            return False
        pass

    def CheckEarlyStop(self):
        if self._indicator_getter is None:
            ValAccLogger.fatal(Fore.LIGHTRED_EX +
                               'you should set the indicator getter by using the method '
                               'SetIndicatorGet'
                               + Fore.RESET)
            sys.exit()
        else:
            pass
        if self._decider is None:
            ValAccLogger.fatal(Fore.LIGHTRED_EX +
                               'you should set the decider by using the method '
                               'SetDecider or use UseDefaultDecider'
                               + Fore.RESET)
            sys.exit()
        else:
            pass
        if self._threshold is None:
            ValAccLogger.fatal(
                Fore.LIGHTRED_EX +
                'you should set the threshold in the UseDefaultDecider method'
                + Fore.RESET
            )
        else:
            pass
        if self._interval is None:
            ValAccLogger.fatal(
                Fore.LIGHTRED_EX +
                'you should set the interval in the UseDefaultDecider method'
                +Fore.RESET
            )
        else:
            pass
        return self
        pass
