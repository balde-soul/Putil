# coding='utf-8'
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
    '--test_time_cost',
    action='store_true',
    default=False,
    dest='TestTimeCost',
    help='set this flag while you want to activate time calculate'
)
parser.add_option(
    '--times',
    action='store',
    default=1,
    dest='Times',
    help='set loop times'
)
parser.add_option(
    '--test_yolo2_build',
    action='store_true',
    default=False,
    dest='TestYolo2Build',
    help='set this flag while you want to test TestYolo2Build'
)
parser.add_option(
    '--test___find_same_cell_location',
    action='store_true',
    default=False,
    dest='TestFindSameCellLocation',
    help='set this flag while you want to test FindSameCellLocation'
)
parser.add_option(
    '--test__standard_yolo2_generate_result',
    action='store_true',
    default=False,
    dest='TestStandardYolo2GenerateResult',
    help='set this flag while you want to test StandardYolo2GenerateResult'
)
parser.add_option(
    '--test__standard_yolo2_generate_feed',
    action='store_true',
    default=False,
    dest='TestStandardYolo2GenerateFeed',
    help='set this flag while you want to test StandardYolo2GenerateFeed'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('test/Yolo/test_yolo2_model_base').logger()
root_logger.setLevel(plog.DEBUG)

TestYolo2BuildLogger = root_logger.getChild('TestYolo2Build')
TestYolo2BuildLogger.setLevel(plog.DEBUG)

TestStandardYolo2GenerateFeedLogger = root_logger.getChild('TestStandardYolo2GenerateFeed')
TestStandardYolo2GenerateFeedLogger.setLevel(plog.DEBUG)

TestStandardYolo2GenerateResultLogger = root_logger.getChild('TestStandardYolo2GenerateResult')
TestStandardYolo2GenerateResultLogger.setLevel(plog.DEBUG)

import Yolo.yolo2_model_base as yolo2b
import numpy as np
import tensorflow as tf
import time

set_up_time_cost = None
times = None


def __tes_yolo2_build():
    TestYolo2BuildLogger.info(th.information(0, 'start testing Yolo2Build', Fore.GREEN) + Fore.RESET)
    feature_feed = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 10], name='other_net_feature')
    f = np.zeros([1, 4, 4, 10], np.float32)
    cl = np.zeros([1, 4, 4, 2], np.int64)
    y = np.zeros([1, 4, 4, 2], np.float32)
    y[0, 2, 2, :] = 64
    x = np.zeros([1, 4, 4, 2], np.float32)
    x[0, 2, 2, :] = 64
    h = np.zeros([1, 4, 4, 2], np.float32)
    h[0, 2, 2] = [10, 5]
    w = np.zeros([1, 4, 4, 2], np.float32)
    w[0, 2, 2] = [2, 3]
    anchor_mask = np.zeros([1, 4, 4, 2], np.float32)
    anchor_mask[0, 2, 2, :] = 1
    build = yolo2b.Yolo2Build(feature_feed, 2, [10, 5], [2, 3], 32, 0.32)
    tf.summary.FileWriter('./model_base-', tf.Session().graph).close()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(
        [build.AnchorLoss, build.PrecisionLoss, build.ClassLoss, build.IndicatorIoU, build.IndicatorClassifyTopOneAcc,
         build.ClassPro, build.GtOneHotClass],
        feed_dict={
            feature_feed: f,
            build.GtClassFeed: cl,
            build.GtYFeed: y,
            build.GtxFeed: x,
            build.GtHFeed: h,
            build.GtWFeed: w,
            build.AnchorMask: anchor_mask
        })

    TestYolo2BuildLogger.debug(
        'anchor loss: {0}, precision loss: {1}, class loss: {2}, indicator_iou: {3}, indicator_classify_top_one_acc: {4}'.format(
            result[0], result[1], result[2], result[3], result[4]))
    if result[0] == 0:
        if result[1] == 1.0:
            TestYolo2BuildLogger.info(
                th.information(0, 'test Yolo2Build successful', Fore.LIGHTGREEN_EX) + Fore.RESET)
            return None
            # if round(result[2], 7) == round(-np.log(0.5), 7):
            #     TestYolo2BuildLogger.info(
            #         th.information(0, 'test Yolo2Build successful', Fore.LIGHTGREEN_EX) + Fore.RESET)
            #     return None
            #     pass
            # else:
            #     TestYolo2BuildLogger.info(
            #         th.information(1, 'class loss should be {0}, but {1}'.format(round(-np.log(0.5), 7),
            #                                                                      round(result[2], 7)),
            #                        Fore.LIGHTRED_EX) + Fore.RESET)
            #     pass
            pass
        else:
            TestYolo2BuildLogger.info(th.information(1, 'precision loss should be {0}, but {1}'.format(1.0, result[1]),
                                                     Fore.LIGHTRED_EX) + Fore.RESET)
            pass
        pass
    else:
        TestYolo2BuildLogger.info(th.information(1, 'anchor loss should be {0}, but {1}'.format(0.0, result[0]),
                                                 Fore.LIGHTRED_EX) + Fore.RESET)
        pass
    TestYolo2BuildLogger.info(th.information(0, 'test Yolo2Build failed', Fore.LIGHTGREEN_EX) + Fore.RESET)
    pass


def __test___find_same_cell_location():
    scalar = 10
    gt_box = [[0, 0, 0, 0], [9, 9, 0, 0], [10, 10, 0, 0], [20, 20, 0, 0]]
    classify = [1, 2, 3, 4]
    base = yolo2b.StandardYolo2Generate([[1, 2]], 32, 0.32)
    format = base.FindSameCellLocation(scalar, gt_box, classify)
    pass


def __test__standard_yolo2_generate_feed():
    TestStandardYolo2GenerateFeedLogger.info(th.information(0, 'start test__standard_yolo2_generate_feed', Fore.LIGHTGREEN_EX) + Fore.RESET)
    prior_hw = [[100, 100], [200, 200]]
    scalar = 32
    _dtype = 0.32
    yolo2_gen = yolo2b.StandardYolo2Generate(prior_hw, scalar)
    feed_height = 32
    feed_width = 32
    gt_box = [[50, 50, 100, 100], [256, 256, 200, 200], [40, 40, 100, 100]]
    class_label = [1, 2, 3]
    # gt_box_0 and gt_box_2 in the same cell: cell: [1, 1] offset:
    feed = yolo2_gen.GenerateFeed(
        {'gt_box': gt_box, 'feed_height': feed_height, 'feed_width': feed_width, 'class': class_label, '_dtype': _dtype})
    y = np.zeros([1, 32, 32, 2], np.float32)
    x = np.zeros([1, 32, 32, 2], np.float32)
    h = np.zeros([1, 32, 32, 2], np.float32)
    w = np.zeros([1, 32, 32, 2], np.float32)
    anchor_mask = np.zeros([1, 32, 32, 2], dtype=np.float32)
    classify = np.zeros([1, 32, 32, 2], np.int32)

    y[0, 1, 1, :] = [40, 50]
    x[0, 1, 1, :] = [40, 50]
    h[0, 1, 1, :] = [100, 100]
    w[0, 1, 1, :] = [100, 100]
    anchor_mask[0, 1, 1, :] = [1, 1]
    classify[0, 1, 1, :] = [3, 1]

    y[0, 8, 8, 1] = 256
    x[0, 8, 8, 1] = 256
    h[0, 8, 8, 1] = 200
    w[0, 8, 8, 1] = 200
    anchor_mask[0, 8, 8, 1] = 1
    classify[0, 8, 8, 1] = 2
    if x.all() == feed['x'].all():
        if y.all() == feed['y'].all():
            if h.all() == feed['h'].all():
                if w.all() == feed['w'].all():
                    if anchor_mask.all() == feed['anchor_mask'].all():
                        if classify.all() == feed['class'].all():
                            TestStandardYolo2GenerateFeedLogger.info(
                                th.information(0, 'test__standard_yolo2_generate_feed successful',
                                               Fore.LIGHTGREEN_EX) + Fore.RESET)
                            pass
                        pass
                    pass
                pass
            pass
        pass
    pass


def __result_check(result_check, i, result_standard_y, result_standard_x, result_standard_h, result_standard_w, result_standard_class):
    if round(result_check['y'], 1) == result_standard_y[i]:
        if round(result_check['x'], 1) == result_standard_x[i]:
            if round(result_check['h'], 1) == result_standard_h[i]:
                if round(result_check['w'], 1) == result_standard_w[i]:
                    if result_check['class'][0] == result_standard_class[i]:
                        return True
                        pass
                    else:
                        TestStandardYolo2GenerateResultLogger.info(
                            th.information(1, 'class supports to be {0}, but {1}'.format(result_standard_class[i],
                                                                                         result_check['class']),
                                           Fore.LIGHTRED_EX) + Fore.RESET)
                        pass
                    pass
                else:
                    TestStandardYolo2GenerateResultLogger.info(
                        th.information(1,
                                       'w supported to be {0}, but {1}'.format(result_standard_w[i],
                                                                               round(result_check['w'], 1)),
                                                                               Fore.LIGHTRED_EX) + Fore.RESET)
                    pass
                pass
            else:
                TestStandardYolo2GenerateResultLogger.info(
                    th.information(1, 'h supported to be {0}, but {1}'.format(result_standard_h[i],
                                                                              round(result_check['h'], 1)),
                                   Fore.LIGHTRED_EX) + Fore.RESET)
                pass
            pass
        else:
            TestStandardYolo2GenerateResultLogger.info(
                th.information(1, 'x supported to be {0}, but {1}'.format(result_standard_x[i],
                                                                          round(result_check['x'], 1)),
                               Fore.LIGHTRED_EX) + Fore.RESET)
            pass
        pass
    else:
        TestStandardYolo2GenerateResultLogger.info(
            th.information(1,
                           'y supported to be {0}, but {1}'.format(result_standard_y[i], round(result_check['y'], 1)),
                           Fore.LIGHTRED_EX) + Fore.RESET)
        pass
    pass


def __test__standard_yolo2_generate_result():
    global times, set_up_time_cost
    TestStandardYolo2GenerateResultLogger.info(
        th.information(0, 'start test__standard_yolo2_generate_feed', Fore.LIGHTGREEN_EX) + Fore.RESET)
    syg = yolo2b.StandardYolo2Generate([[10, 10], [20, 20], [30, 30]], 32)
    y = np.zeros([1, 4, 4, 3])
    x = np.zeros([1, 4, 4, 3])
    h = np.zeros([1, 4, 4, 3])
    w = np.zeros([1, 4, 4, 3])
    precision = np.zeros([1, 4, 4, 3])
    class_pro = np.zeros([48, 3])

    precision[:, 2, 2, :] = [0.5, 0.6, 0.7]
    y[:, 2, 2, :] = [0.1, 0.5, 0.7]
    x[:, 2, 2, :] = [0.1, 0.5, 0.7]
    h[:, 2, 2, :] = [0.5, 1.0, 1.5]
    w[:, 2, 2, :] = [0.5, 1.0, 1.5]
    class_pro[14:17:1, :] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    result_standard_y = [67.2, 80.0, 86.4]
    result_standard_x = [67.2, 80.0, 86.4]
    result_standard_h = [16.5, 54.4, 134.5]
    result_standard_w = [16.5, 54.4, 134.5]
    result_standard_class = [0, 1, 2]

    if set_up_time_cost:
        start = time.clock()
        for i in range(0, times):
            result = syg.GenerateResult(
                {'y': y, 'x': x, 'h': h, 'w': w, 'precision': precision, 'class': class_pro, 'threshold': 0.65})
            pass
        end = time.clock()
        TestStandardYolo2GenerateResultLogger.info(th.information(1,
                                                                  '{0} circulations cost {1}, every time cost {2}'.format(
                                                                      times, end - start, (end - start) / times),
                                                                      Fore.LIGHTYELLOW_EX) + Fore.RESET)
    else:
        result = syg.GenerateResult({'y': y, 'x': x, 'h': h, 'w': w, 'precision': precision, 'class': class_pro, 'threshold': 0.4})
        check_1 = True
        check_2 = True
        check_3 = True
        for i in result:
            if i['pre'] == 0.5:
                check_1 = __result_check(i, 0, result_standard_y, result_standard_x, result_standard_h,
                                         result_standard_w,
                                         result_standard_class)
            if i['pre'] == 0.6:
                check_2 = __result_check(i, 1, result_standard_y, result_standard_x, result_standard_h,
                                         result_standard_w,
                                         result_standard_class)
                pass
            if i['pre'] == 0.7:
                check_3 = __result_check(i, 2, result_standard_y, result_standard_x, result_standard_h,
                                         result_standard_w,
                                         result_standard_class)
                pass
            pass
        if check_1 and check_2 and check_3:
            TestStandardYolo2GenerateResultLogger.info(
                th.information(0, 'test__standard_yolo2_generate_feed successful', Fore.LIGHTGREEN_EX) + Fore.RESET)
        else:
            TestStandardYolo2GenerateResultLogger.info(
                th.information(0, 'test__standard_yolo2_generate_feed failed', Fore.LIGHTGREEN_EX) + Fore.RESET)
            pass
    pass


if __name__ == '__main__':
    # global set_up_time_cost, times
    set_up_time_cost = options.TestTimeCost
    times = int(options.Times)
    if options.TestYolo2Build:
        __tes_yolo2_build()
        pass
    if options.TestFindSameCellLocation:
        __test___find_same_cell_location()
        pass
    if options.TestStandardYolo2GenerateFeed:
        __test__standard_yolo2_generate_feed()
        pass
    if options.TestStandardYolo2GenerateResult:
        __test__standard_yolo2_generate_result()
        pass
    pass

