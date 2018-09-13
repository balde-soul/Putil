# coding=utf-8
from optparse import OptionParser
import Putil.loger as plog
from colorama import Fore
import Putil.test.test_helper as th
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
    '--test_ValAccStop',
    action='store_true',
    default=False,
    dest='TestValAccStop',
    help='set this flag while you want to test ValAccStop'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('tf/test/tf/param/test_auto_save').logger()
root_logger.setLevel(plog.DEBUG)
TestVallAccStopLogger = root_logger.getChild("TestVallAccStop")
TestVallAccStopLogger.setLevel(plog.DEBUG)

import Putil.tf.param.early_stop as es


class model:
    def __init__(self):
        self._data = [1.0, 2.0, 3.0, 4.0, 5.0,
                      6.0, 7.0, 8.0, 0.0, 0.1,
                      0.4, 0.6, 0.7, 8.5, 9.0,
                      10.0, 10.0, 10.0, 10.0, 10.0,
                      10.0, 10.0, 10.0, 10.0, 10.0]
        self._i = -1
        """
        1: while indicator does not rise up to 1.0 for 5 val epoch, EarlyStop would post a Stop: True
           which means this part would stop in: 12
        """
        self._early_stop_1 = es.ValAccStop().UseDefaultDecider(max=True, interval=5, threshold=1.0).SetIndicatorGet(self.Output).CheckEarlyStop()
        """
        1: while indicator does not rise up to 1.0 for 7 val epoch, EarlyStop would post a Stop: True
           which means this part would stop in: 22
        """
        self._early_stop_2 = es.ValAccStop().UseDefaultDecider(max=True, interval=7, threshold=1.0).SetIndicatorGet(self.Output).CheckEarlyStop()
        pass

    def ModelCheck1(self):
        return self._early_stop_1.Stop()
        pass

    def ModelCheck2(self):
        return self._early_stop_2.Stop()
        pass
    
    def Output(self):
        return self._data[self._i]

    def TrainCV(self):
        self._i += 1
        pass


    pass


def __test_valacc_stop():
    TestVallAccStopLogger.info(th.information(0, 'start test valacc_stop', Fore.GREEN) + Fore.RESET)
    m1 = model()
    stop_one_1_index = 12
    one = 0
    two = 0
    stop_one_2_index = 22
    for i in range(0, 25):
        m1.TrainCV()
        if one == 0:
            if m1.ModelCheck1() is True:
                one = i
            else:
                pass
            pass
        if two == 0:
            if m1.ModelCheck2() is True:
                two = i
                pass
            else:
                pass
            pass
        pass
    if (one == stop_one_1_index and two == stop_one_2_index) is True:
        TestVallAccStopLogger.info(th.information(0, 'test valacc_stop successful', Fore.GREEN) + Fore.RESET)
    else:
        TestVallAccStopLogger.info(th.information(0, 'test valacc_stop failed', Fore.LIGHTRED_EX) + Fore.RESET)
    pass


if __name__ == '__main__':
    if options.TestValAccStop is True:
        __test_valacc_stop()
