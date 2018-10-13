# coding=utf-8
from optparse import OptionParser
import Putil.base.logger as plog
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
    '--test_all',
    action='store_true',
    default=False,
    dest='TestAll',
    help='set this flag while you want to test ALL'
)
parser.add_option(
    '--test_calc_iou',
    action='store_true',
    default=False,
    dest='TestCalcIou',
    help='set this flag while you want to test calc_iou'
)
parser.add_option(
    '--test_calc_iou_matrix_chw',
    action='store_true',
    default=False,
    dest='TestCalcIouMatrixCHW',
    help='set this flag while you want to test calc_iou_matrix_chw'
)
parser.add_option(
    '--test_calc_iou_matrix_thw',
    action='store_true',
    default=False,
    dest='TestCalcIouMatrixTHW',
    help='set this flag while you want to test calc_iou_matrix_thw'
)
parser.add_option(
    '--test_calc_iou_matrix_ohw',
    action='store_true',
    default=False,
    dest='TestCalcIouMatrixOHW',
    help='set this flag while you want to test calc_iou_matrix_ohw'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('test/calc/test_estimate').logger()
root_logger.setLevel(plog.DEBUG)
TestCalcIouLogger = root_logger.getChild('TestCalcIou')
TestCalcIouLogger.setLevel(plog.DEBUG)
TestCalcIouMatrixCHWLogger = root_logger.getChild('TestCalcMatrixCHW')
TestCalcIouMatrixCHWLogger.setLevel(plog.DEBUG)
from colorama import Fore
import numpy as np
import Putil.tf.static as tfs
import Putil.test.test_helper as th
import Putil.calc.estimate as es


def __test_calc_iou():
    TestCalcIouLogger.info(th.information(0, 'test_calc_iou', Fore.GREEN))
    rect1 = [0, 0, 10, 10]
    rect2 = [5, 5, 10, 10]
    rect3 = [75, 75, 100, 100]
    iou = es.calc_iou(rect1, rect2, LHW=True)
    correct = 25.0 / 175
    try:
        assert iou == correct
        TestCalcIouLogger.info(th.information(0, 'test_calc_iou successful', Fore.GREEN))
    except Exception:
        TestCalcIouLogger.info(th.information(1, '>>:should be {0}, but {1}'.format(correct, iou), Fore.LIGHTRED_EX))
    pass


def __test_calc_iou_matrix_chw():
    TestCalcIouLogger.info(th.information(0, 'test_calc_iou_matrix_chw', Fore.GREEN))
    rect1 = [[5, 5, 10, 10], [10, 10, 20, 20]]
    rect2 = [[25, 25, 40, 40], [45, 45, 80, 80]]
    iou = es.calc_iou_matrix_chw(rect1, rect2)
    correct = np.array([[25.0 / 1675, 25.0 / 6475], [225.0 / 1475, 225.0 / 6575]])
    try:
        assert iou.all() == correct.all()
        TestCalcIouLogger.info(th.information(0, 'test_calc_iou_matrix_chw successful', Fore.GREEN))
    except Exception as e:
        TestCalcIouLogger.info(th.information(1, '>>:should be {0}, but {1}'.format(correct, iou), Fore.LIGHTRED_EX))
        pass
    pass


def __test_calc_iou_matrix_thw():
    TestCalcIouLogger.info(th.information(0, 'test_calc_iou_matrix_thw', Fore.GREEN))
    rect1 = [[0, 0, 10, 10], [0, 0, 20, 20]]
    rect2 = [[5, 5, 40, 40], [5, 5, 80, 80]]
    iou = es.calc_iou_matrix_thw(rect1, rect2)
    correct = np.array([[25.0 / 1675, 25.0 / 6475], [225.0 / 1475, 225.0 / 6575]])
    try:
        assert iou.all() == correct.all()
        TestCalcIouLogger.info(th.information(0, 'test_calc_iou_matrix_thw successful', Fore.GREEN))
    except Exception as e:
        TestCalcIouLogger.info(th.information(1, '>>:should be {0}, but {1}'.format(correct, iou), Fore.LIGHTRED_EX))
        pass
    pass


def __test_calc_iou_matrix_ohw():
    TestCalcIouLogger.info(th.information(0, 'test_calc_iou_matrix_ohw', Fore.GREEN))
    rect1 = [[0, 0, 10, 10], [0, 0, 20, 20]]
    rect2 = [[5, 5, 40, 40], [5, 5, 80, 80]]
    iou = es.calc_iou_matrix_thw(rect1, rect2)
    correct = np.array([[25.0 / 1675, 25.0 / 6475], [225.0 / 1475, 225.0 / 6575]])
    try:
        assert iou.all() == correct.all()
        TestCalcIouLogger.info(th.information(0, 'test_calc_iou_matrix_ohw successful', Fore.GREEN))
    except Exception as e:
        TestCalcIouLogger.info(th.information(1, '>>:should be {0}, but {1}'.format(correct, iou), Fore.LIGHTRED_EX))
        pass
    pass


def __test_all():
    __test_calc_iou()
    pass


if __name__ == '__main__':
    if options.TestAll:
        __test_all()
        __test_calc_iou_matrix_chw()
        __test_calc_iou_matrix_thw()
        __test_calc_iou_matrix_ohw()
        pass
    else:
        if options.TestCalcIou:
            __test_calc_iou()
            pass
        if options.TestCalcIouMatrixCHW:
            __test_calc_iou_matrix_chw()
            pass
        if options.TestCalcIouMatrixTHW:
            __test_calc_iou_matrix_thw()
            pass
        if options.TestCalcIouMatrixOHW:
            __test_calc_iou_matrix_ohw()
            pass

