from optparse import OptionParser
import Putil.loger as plog
from colorama import Fore, Style
parser = OptionParser(usage='usage %prog [options] arg1 arg2')
parser.add_option(
    '--test_dict_pile',
    action='store_true',
    dest='TestDictPile',
    default=False,
    help='while you want to test DictPile , set this flag'
)
parser.add_option(
    '--test_logger_base',
    action='store_true',
    dest='TestLoggerBase',
    default=False,
    help='while you want to test LoggerBase , set this flag'
)
level_default = 'Info'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig("tf/base/test_logger_baser").logger()
root_logger.setLevel(plog.DEBUG)
TestDictPileLogger = root_logger.getChild('TestDictPile')
TestDictPileLogger.setLevel(plog.DEBUG)
TestLoggerBaserLogger = root_logger.getChild('TestLoggerBaser')
TestLoggerBaserLogger.setLevel(plog.DEBUG)

import Putil.base.logger_base as lb


def __test_dict_pile():
    a = {
        '0': 1,
        '1': {
            '0': 2,
            '1': 2,
            '2': {
                '0': 3,
                '1': 3
            }
        },
        '2': [1, 2, 3]
    }
    info = lb.DictPile(a, pre_info='\n')
    TestDictPileLogger.info(Fore.LIGHTGREEN_EX + info + Fore.RESET)
    pass


def __test_logger_base():
    a = {
        '0': 1,
        '1': {
            '0': 2,
            '1': 2,
            '2': {
                '0': 3,
                '1': 3
            }
        },
        '2': [1, 2, 3]
    }
    lb.LoggerBase(TestLoggerBaserLogger).DictLog(a, plog.INFO, Fore.LIGHTGREEN_EX, pre_info='\n')
    pass


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    if options.TestDictPile:
        __test_dict_pile()
        pass
    if options.TestLoggerBase:
        __test_logger_base()
        pass
    else:
        pass
    pass
