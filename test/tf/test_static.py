# coding=utf-8
from optparse import OptionParser
import Putil.loger as plog
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
parser.add_option(
    '--test_param_probe',
    action='store_true',
    default=False,
    dest='TestParamProbe',
    help='set this flag while you want to test ParamProbe'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")
from colorama import Fore
import Putil.tf.static as tfs
import Putil.test.test_helper as th


root_logger = plog.PutilLogConfig("tf/test/test_static").logger()
root_logger.setLevel(plog.DEBUG)
TestParamProbeLogger = root_logger.getChild("TestParamProbe")
TestParamProbeLogger.setLevel(plog.DEBUG)


def __test_param_probe():
    TestParamProbeLogger.info(th.information(0, 'start test param probe', Fore.GREEN))
    default = {'a': [1, 2, 3], 'k': {'l': 1, 'm': {'n': 4}}, 'b': {'c': 'sdsd', 'd': False, 'e': {'f': 1.0, 'g': [1, 2], 'h': {'i': 1}}}}
    feed = {'b': {'e': {'h': {}}}, 'k': {'m': {}}}
    feed = tfs.ParamProbe(default, feed).fix_with_default().ParamGenWithInfo(TestParamProbeLogger)
    try:
        assert feed == default
        TestParamProbeLogger.info(th.information(0, "param probe successful", Fore.GREEN))
    except AssertionError as e:
        TestParamProbeLogger.info(th.information(0, "param probe failed", Fore.RED))
        TestParamProbeLogger.debug(th.information(1, e, Fore.RED))
        pass
    pass


if __name__ == '__main__':

    if options.TestParamProbe:
        __test_param_probe()
    else:
        pass
    pass
