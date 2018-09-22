# coding=utf-8
from optparse import OptionParser
import Putil.loger as plog
from colorama import Fore
import functools
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
    '--test_ImproveSave',
    action='store_true',
    default=False,
    dest='TestImproveSave',
    help='set this flag while you want to test ImproveSave'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('tf/test/tf/param/test_auto_save').logger()
root_logger.setLevel(plog.DEBUG)

TestImproveSaveLogger = root_logger.getChild('TestImproveSave')
TestImproveSaveLogger.setLevel(plog.DEBUG)

import Putil.tf.param.auto_save as auto_save


class model:
    def __init__(self):
        self._auto_save = auto_save.ImproveSave(5).UseDefaultDecider(max=True).SetIndicatorGet(self.Output).CheckAutoSave()
        self._data = [0.0, 1.0, 1.2, 1.3, 1.4, 1.1, 1.3, 1.7, 1.9, 1.0, 1.5]
        self._i = -1
        pass

    def ModelCheck(self):
        if self._auto_save.Save() is True:
            TestImproveSaveLogger.debug('save in acc: {0}'.format(self.Output()))
            return True
        else:
            return False
        pass

    @property
    def AutoSave(self):
        return self._auto_save

    def TrainCv(self):
        self._i += 1
        pass

    @property
    def Data(self):
        return self._data

    def Output(self):
        return self.Data[self._i]


def __test_improve_save():
    print(th.information(0, 'start testing imrpove_save', Fore.GREEN) + Fore.RESET)
    m = model()
    print(m._auto_save._regular)
    assert m._auto_save.IndicatorGetter == m.Output
    assert m._auto_save.DecisionGenerator == m._auto_save._decider
    hit_target = [1, 2, 3, 4, 7, 8]
    for i in range(0, 11):
        m.TrainCv()
        if m.ModelCheck() is True:
            if i in hit_target:
                pass
            else:
                print(th.information(0, 'test improve_save failed', Fore.LIGHTRED_EX) + Fore.RESET)
                pass
            pass
        pass
    print(th.information(0, 'test improve_save successful', Fore.LIGHTGREEN_EX) + Fore.RESET)
    pass


if __name__ == '__main__':
    if options.TestImproveSave:
        __test_improve_save()

