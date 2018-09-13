# coding=utf-8
import abc
import six
import Putil.loger as plog
import sys
from colorama import Fore

root_logger = plog.PutilLogConfig('tf/param/learning_update').logger()
root_logger.setLevel(plog.DEBUG)
ReduceROnPlateauClassLogger = root_logger.getChild('ReduceROnPlateauClass')
ReduceROnPlateauClassLogger.setLevel(plog.DEBUG)
TrainIndicatorLogger = root_logger.getChild('TrainIndicator')
TrainIndicatorLogger.setLevel(plog.DEBUG)

"""
                        common api
                             |
                        abs ReduceLROnPlateau
                             |
                        menage by param
                             |
     --------------------------------------------------
    |                         |                |       |
    |                         |                |       |
ReduceLROnPlateau_1    ReduceLROnPlateau_2    ....    ....
"""
"""
ReduceLROnPlateau usage:
    Generator ReduceLROnPlateau in Model class ,and use it while in training
"""


@six.add_metaclass(abc.ABCMeta)
class ReduceLROnPlateau:
    @abc.abstractmethod
    def Reduce(self):
        """
        an abstract method which return the bool which represent the flag of reduce or not
        True means: reduce learning rate is post
        False means: not to reduce learning rate is post
        :return: bool
        """
        pass

    @abc.abstractmethod
    def SetDecider(self, decider):
        """
        an bastract method which provide way to set the decider
        :param decider: function
        :return:
        """
        pass

    @abc.abstractmethod
    def SetIndicatorGet(self, getter):
        """
        an abstract method which provide the method to set the indicator getter
        :param getter:
        :return:
        """
        pass

    @abc.abstractmethod
    def CheckReduceLROnPlateau(self):
        """
        an abstract method to check the object is OK or Not:
        :return:
        """
        pass
    pass


"""
    this class is use for generator a new ReduceLROnPlateau object
"""


class ReduceROnPlateauClass(ReduceLROnPlateau):
    def __init__(self):
        self._decider = None
        self._indicator_getter = None
        pass

    def Reduce(self):
        return self._decider(self._indicator_getter())

    def CheckReduceLROnPlateau(self):
        if self._decider is None:
            ReduceROnPlateauClassLogger.fatal(Fore.LIGHTRED_EX + 'you should set the decider by using the method'
                                                                 'SetDecider' + Fore.RESET)
            sys.exit()
        else:
            pass
        if self._indicator_getter is None:
            ReduceROnPlateauClassLogger.fatal(Fore.LIGHTRED_EX + 'you should set the indicator getter by usiing the '
                                                                 'method SetIndicatorGet' + Fore.RESET)
            sys.exit()
            pass
        else:
            pass
        return self
        pass

    def SetIndicatorGet(self, getter):
        self._indicator_getter = getter
        return self
        pass

    def SetDecider(self, decider):
        self._decider = decider
        return self
        pass
    pass


"""
usage:this class provide a more sample way to apply LrUpdate , is frequently-used
    check in the test program
class model:
    def __init__(self):
        self._val_lr_update = LrUpdate().UseDefaultDecider(max=True, interval=5, epsilon=1.0, cooldown=3).SetIndicatorGet(self.Output).CheckReduceLROnPlateau()

    @property
    def Output(self):
        return self._data[self._i]
        pass
"""

# tested


class LrUpdate(ReduceROnPlateauClass):
    """provide the default method"""
    def __init__(self):
        ReduceROnPlateauClass.__init__(self)
        self._epsilon = None
        self._interval = None
        self._regular = None
        self._cooldown = None
        self._cooldown_count = 0
        # self._record = list()
        self._best = None
        self._interval_count = 0
        self._max = None
        self._min = None
        pass

    def UseDefaultDecider(self, **options):
        """
        set the Decider parameter
        this class provide a frequent-used Decider(while no improved up to epsilon for n epoch val
            Stop while return a True, in this way, 'improve' could be indicator rise or indicator descend,
            once Reduce return True, this object would cool down for cool_down val epoch
        :param options:
            epsilon: the threshold which decide improve woork or not
            interval: patience for interval val epoch
            cooldown: cool down the reduce object for cooldown epoch
            max: treats indicator rise as improved
            min: treats indicator descend as improved
            *max and min can not be same at the same time*
        :return:
        """
        self._interval = options.pop('interval', self._interval)
        self._epsilon = options.pop('epsilon', self._epsilon)
        self._cooldown = options.pop('cooldown', self._cooldown)
        self._max = options.pop('max', self._max)
        self._min = options.pop('min', self._min)
        if self._max == self._min:
            TrainIndicatorLogger.fatal(
                Fore.LIGHTGREEN_EX +
                'max and min should not be True at the same time '
                '\nyou should set min or max to True'
                + Fore.RESET)
            pass
        self._regular = True if self._max is True else self._regular
        self._regular = False if self._min is True else self._regular
        self.SetDecider(self._default_decider)
        return self
        pass

    def _default_decider(self, indicator):
        if self._cooldown_count <= self._cooldown:
            self._cooldown_count += 1
            ret = False
            pass
        else:
            if self._best is None:
                self._best = indicator
                ret = False
                pass
            else:
                if (indicator - self._best >= self._epsilon) == self._regular:
                    self._interval_count = 0
                    self._best = indicator
                else:
                    self._interval_count += 1
                    pass
                if self._interval_count >= self._interval:
                    self._interval_count = 0
                    TrainIndicatorLogger.info(
                        Fore.LIGHTGREEN_EX +
                        'learning rate reduce:'
                        '\nregular: {0}'
                        '\ninterval: {1}'
                        '\ncooldown: {2}'
                        '\nepsilon: {3}'.format(
                            'max' if self._regular is True else 'min',
                            self._interval,
                            self._cooldown,
                            self._epsilon
                        )
                        + Fore.RESET)
                    ret = True
                    pass
                else:
                    ret = False
                    pass
                pass
            pass
        return ret
        pass

