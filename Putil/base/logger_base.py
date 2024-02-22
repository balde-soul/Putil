# coding=utf-8
import Putil.base.logger as plog
from colorama import Fore, Style, init

root_logger = plog.PutilLogConfig('base/logger_base').logger()
root_logger.setLevel(plog.DEBUG)


# DEBUG = plog.DEBUG
# INFO = plog.INFO
# ERROR = plog.ERROR
# FATAl = plog.FATAL
#
# COLOR = Fore

DictReflect = lambda key, value, info, indent: '{0}\n{3}{1}:  {2}'.format(info, key, value, ' ' * indent) \
        if info != '' else '{3}{0}{1}:  {2}'.format(info, key, value, ' ' * indent)


def DictPile(_dict, pre_info='', indent=4, step=0):
    """
    this function is used to print dict information in standard format
    :param _dict: wanted to be printed
    :param pre_info: information to be added before the dict print part
    :param indent: indent
    :param step: base step for indent
    :return:
    """
    for i in _dict.keys():
        if type(_dict[i]).__name__ == 'dict':
            pre_info = '{0}\n{2}{1}:  '.format(pre_info, i, ' ' * indent * step) \
                if pre_info != '' else '{0}{2}{1}:  '.format(pre_info, i, ' ' * indent * step)
            step += 1
            pre_info = DictPile(_dict[i], pre_info, indent, step)
            step -= 1
            pass
        else:
            pre_info = DictReflect(i, _dict[i], pre_info, indent * step)
            pass
        pass
    return pre_info


"""
this class is aimed at collection different data type print format
such as Dict or else,
waiting to complement
"""


class LoggerBase:
    def __init__(self, logger):
        """
        init the object
        :param logger: the logger which printing the information
        """
        self._logger = logger
        pass

    def DictLog(self, _dict, level=plog.DEBUG, color=Fore.RESET, pre_info='', indent=4, step=0):
        plog.Log(self._logger, level)(color + DictPile(_dict, pre_info, indent, step) + Fore.RESET)
        return self
        pass

    def SetLogger(self, logger):
        self._logger = logger
        return self
    pass


