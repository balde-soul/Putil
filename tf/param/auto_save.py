# coding=utf-8
import Putil.loger as plog
from colorama import Fore, init, Style
import sys
import six
import abc

root_logger = plog.PutilLogConfig('tf/param/auto_save').logger()
root_logger.setLevel(plog.DEBUG)

AutoSaveClassLogger = root_logger.getChild('AutoSaveClass')
AutoSaveClassLogger.setLevel(plog.DEBUG)

ImproveSaveLogger = root_logger.getChild('ImproveSave')
ImproveSaveLogger.setLevel(plog.DEBUG)

"""
                common api
                     |
                abs AutoSave
                     |
                menage by param
                     |
     ------------------------------------
    |                |           |       |
    |                |           |       |
auto_save_1    auto_save_2    ....    ....
"""
"""
AutoSave usage:
    Generator AutoSave in Model class ,and use it while in training

"""


@six.add_metaclass(abc.ABCMeta)
class AutoSave(object):
    @abc.abstractmethod
    def Save(self):
        """
        get the flag for save the weight or not
        :return: bool True represent suppose to save the weight, False represent not suppose to save the weight
        """
        pass

    @abc.abstractmethod
    def CheckAutoSave(self):
        """
        an abstract method to check the object is Ok or not
        :return: bool
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
    def SetDecider(self, decider):
        """
        an abastract method which provide way to set the decider
        :param daecider: function
        :return:self
        """
        pass
    pass


"""
    this class is use for generator a new EarlyStop object
"""


@six.add_metaclass(abc.ABCMeta)
class AutoSaveClass(AutoSave):
    def __init__(self):
        self._decider = None
        self._indicator_getter = None
        pass

    def Save(self):
        return self._decider(self._indicator_getter())
        pass

    def CheckAutoSave(self):
        if self._indicator_getter is None:
            AutoSaveClassLogger.fatal(Fore.LIGHTRED_EX +
                                      'you should set the indicator getter by using the method '
                                      'SetIndicatorGet'
                                      + Fore.RESET
                                      )
            sys.exit()
        else:
            pass
        if self._decider is None:
            AutoSaveClassLogger.fatal(Fore.LIGHTRED_EX +
                                      'you should set the decider by using the method '
                                      'SetDecider'
                                      + Fore.RESET
                                      )
            sys.exit()
        else:
            pass
        return self

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
this class provide some methods for auto save the model in effective epoch
according to a indicator, while the indicator improve , save the model
basic usage:
use default decider
class model:
    def __init__(self):
        self._acc = None
        self._auto_save = ImproveSave(5).UseDefaultDecider(wanted_maximum=True).SetIndicatorGet(self._get_acc)
        pass

    def _get_acc(self):
        return self.Output['acc']
        pass

    @property
    def Output(self):
        return { 'acc': self._acc}
        pass
usage personal decider
class model:
    def __init__(self):
        self._acc = None
        self._auto_save = ImproveSave(5).SetDicider(self._decider).SetIndicatorGet(self._get_acc)
        pass

    def _get_acc(self):
        return self.Output['acc']
        pass

    def _decider(self, acc):
        ...
        pass

    @property
    def Output(self):
        return { 'acc': self._acc}
        pass

"""


class ImproveSave(AutoSaveClass):
    def __init__(self, model_keep_amount):
        AutoSaveClass.__init__(self)
        # the number of model keep
        self._model_keep_amount = model_keep_amount
        # the saved weight indicator which is the same length of the self._model_keep_amount
        self._best_record = list()
        self._regular = None
        pass

    def CheckAutoSave(self):
        """
        check the save can work or not
        :return: bool, True: model can save the weight. False: model is not necessary to save the weight(decide by the
        self._decision_generator
        """
        if self._decider is None:
            ImproveSaveLogger.fatal(Fore.LIGHTRED_EX + 'have not set decision_generator which is the method for '
                                                       'decide save the model weight or not , if you have no plane to'
                                                       'complement the method you can use the default decision by call'
                                                       'the method UseDefaultDecider which suggest save the weight will'
                                                       'the indicator improve, indicator can the acc, etc.' +
                                    Fore.RESET)
            sys.exit()
            pass
        if self._indicator_getter is None:
            ImproveSaveLogger.fatal(Fore.LIGHTRED_EX + 'have not set the indicator_get which is the method to get the '
                                                       'indicator from the model, there is no way to provide the '
                                                       'default method for get the indicator , because it is depend on '
                                                       'the model you should provide the method by yourself' +
                                    Fore.RESET)
            sys.exit()
            pass
        return self
        pass

    # def SaveFlage(self):
    #     """
    #     this is the method which model call to check save or not
    #     :return: bool, True: model can save the weight. False: model is not necessary to save the weight(decide by the
    #     self._decision_generator
    #     """
    #     return self._decision_generator(self._indicator_get())
    #     pass

    def UseDefaultDecider(self, **options):
        """
        the default decider, while the indicator improve(lower or bigger), the decider would return the True
        :param options:
            max and min can not be setted in the sane time
            :param max: the flag which represent we want the indicator to be maximum
            :param min: the flag which represent we want the indicator to be minimum
        :return:
        """
        ImproveSaveLogger.info(options)
        max = options.pop('max', None)
        min = options.pop('min', None)
        if max == min:
            ImproveSaveLogger.fatal(Fore.LIGHTGREEN_EX +
                                    'max and min should not be True at the same time'
                                    + Fore.RESET
                                    )
            raise ValueError()
        self._regular = True if max is True else self._regular
        self._regular = False if min is True else self._regular
        if self._regular is True:
            self._decider = self.__default_decider_wanted_maximum
            pass
        else:
            self._decider = self.__default_decider_wanted_minimum
            pass
        return self
        pass

    def __default_decider_wanted_minimum(self, indicator):
        """
        the default decider which want the indicator to be min
        :param indicator:
        :return:
        """
        if len(self._best_record) != 0:
            if indicator < self._best_record[0]:
                ImproveSaveLogger.info(Fore.LIGHTGREEN_EX + 'indicator reduce from {0} to {1}, '
                                                            'set the ask to save the model weight'
                                                            ''.format(
                    self._best_record, indicator
                ) + Fore.RESET)
                if len(self._best_record) < self._model_keep_amount:
                    self._best_record.insert(0, indicator)
                else:
                    self._best_record.pop()
                    self._best_record.insert(0, indicator)
                    pass
                return True
                pass
            else:
                return False
                pass
        else:
            self._best_record.insert(0, indicator)
        pass

    def __default_decider_wanted_maximum(self, indicator):
        """
        the default decider which want the indicator to be max
        :param indicator: the indicator which should be a value
        :return:
        """
        if len(self._best_record) != 0:
            if indicator > self._best_record[0]:
                ImproveSaveLogger.info(Fore.LIGHTGREEN_EX + 'indicator improve from {0} to {1}, '
                                                            'set the ask to save the model weight'
                                                            ''.format(
                    self._best_record[0], indicator
                ) + Fore.RESET)
                if len(self._best_record) < self._model_keep_amount:
                    self._best_record.insert(0, indicator)
                else:
                    self._best_record.pop()
                    self._best_record.insert(0, indicator)
                    pass
                return True
                pass
            else:
                return False
                pass
        else:
            self._best_record.insert(0, indicator)
        pass

    @property
    def IndicatorGetter(self):
        """
        get the indicator getter
        :return:
        """
        return self._indicator_get
        pass

    @property
    def DecisionGenerator(self):
        """
        get the decision_generator
        :return:
        """
        return self._decision_generator
    pass

