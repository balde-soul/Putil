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

import Yolo.yolo2_model_base as yolo2b
import numpy as np
import tensorflow as tf


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
    result = sess.run([build.AnchorLoss, build.PrecisionLoss, build.ClassLoss, build.ClassPro, build.GtOneHotClass],
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
        'anchor loss: {0}, precision loss: {1}, class loss: {2}'.format(result[0], result[1], result[2]))
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
    yolo2_gen = yolo2b.StandardYolo2Generate(prior_hw, scalar, _dtype)
    feed_height = 32
    feed_width = 32
    gt_box = [[50, 50, 100, 100], [256, 256, 200, 200], [40, 40, 100, 100]]
    class_label = [1, 2, 3]
    # gt_box_0 and gt_box_2 in the same cell: cell: [1, 1] offset:
    feed = yolo2_gen.GenerateFeed(
        {'gt_box': gt_box, 'feed_height': feed_height, 'feed_width': feed_width, 'class': class_label})
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


if __name__ == '__main__':
    if options.TestYolo2Build:
        __tes_yolo2_build()
        pass
    if options.TestFindSameCellLocation:
        __test___find_same_cell_location()
        pass
    if options.TestStandardYolo2GenerateFeed:
        __test__standard_yolo2_generate_feed()
    pass

