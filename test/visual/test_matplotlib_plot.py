# coding=utf-8
from optparse import OptionParser
parser = OptionParser(usage='usage %prog [options] arg1 arg2')

level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
random_type_default = False
parser.add_option(
    '--random_type',
    action='store_true',
    dest='RandomType',
    # type=bool,
    default=random_type_default,
    help='specify the flag if you want to test the random_type'
         'default: {0}'.format(random_type_default)
)

import Putil.base.logger as plog
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")
root_logger = plog.PutilLogConfig('test/Yolo/test_yolo2_model_base').logger()
root_logger.setLevel(plog.DEBUG)

TestRandomTypeLogger = root_logger.getChild('TestRnadomType').logger()
TestRandomTypeLogger.setLevel(plog.DEBUG)


from Putil.visual.matplotlib_plot import random_type
import test.test_helper as th
from colorama import Fore


def __test_random_type():
    TestRandomTypeLogger.info(th.information(0, 'start test random_type', Fore.LIGHTGREEN_EX))

    def check_log(target, standard, get):
        TestRandomTypeLogger.info(th.information(1, '{2} should be {0} but {1}'.format(
            standard, get, target), Fore.LIGHTRED_EX))
        flag = False
        pass

    flag = True
    type = random_type()
    for i in range(0, 20):
        _type = type.type_gen(color='r')
        None if _type[0] == 'r' else check_log('_type[0]', 'r', _type[0])
        pass
    for i in range(0, 20):
        _type = type.type_gen(marker='o')
        None if _type[0] == 'o' else check_log('_tyupe[0]', 'o', _type[0])
        pass
    for i in range(0, 20):
        _type = type.type_gen(line='--')
        None if _type[2:] == '--' else check_log('_type[2:]', '--', _type[2:])
        pass
    TestRandomTypeLogger.info(th.information(0, 'test random_type successful', Fore.LIGHTGREEN_EX)) if flag \
    else TestRandomTypeLogger.info(th.information(0, 'test random_type failed', Fore.LIGHTRED_EX))
    pass


def __test():
    __test_random_type() if options.RandomType else None


if __name__ == '__main__':
    __test()
    pass
