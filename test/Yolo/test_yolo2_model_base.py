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
    help='set this flag while you want to test TestFindSameCellLocation'
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

import Yolo.yolo2_model_base as yolo2b
import numpy as np
import tensorflow as tf


def __tes_yolo2_build():
    print(th.information(0, 'start testing Yolo2Build', Fore.GREEN) + Fore.RESET)
    feature_feed = tf.placeholder(dtype=tf.float32, shape=[10, 10, 10, 100], name='other_net_feature')
    f = np.zeros([10, 10, 10, 100], np.float32)
    p_mask = np.zeros([10, 10, 10, 1], np.float32)
    n_mask = np.ones([10, 10, 10, 1], np.float32)
    cl = np.zeros([10, 10, 10, 4], np.int64)
    y = np.zeros([10, 10, 10, 4], np.float32)
    x = np.zeros([10, 10, 10, 4], np.float32)
    h = np.zeros([10, 10, 10, 4], np.float32)
    w = np.zeros([10, 10, 10, 4], np.float32)
    anchor_mask = np.zeros([10, 10, 10, 4], np.float32)
    yolo_feature = yolo2b.gen_pro(feature_feed, 3, 4)
    loss, place = yolo2b.append_yolo2_loss(yolo_feature, 3, [10, 5, 3, 4], [2, 3, 4, 8], 32)
    tf.summary.FileWriter('./model_base-', tf.Session().graph).close()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([loss], feed_dict={
        feature_feed: f,
        place['class']: cl,
        place['y']: y,
        place['x']: x,
        place['h']: h,
        place['w']: w,
        place['p_mask']: p_mask,
        place['n_mask']: n_mask,
        place['anchor_mask']: anchor_mask
    }))
    print(th.information(0, 'test Yolo2Build successful', Fore.LIGHTGREEN_EX) + Fore.RESET)
    pass


def __test___find_same_cell_location():
    scalar = 10
    gt_box = [[0, 0, 0, 0], [9, 9, 0, 0], [10, 10, 0, 0], [20, 20, 0, 0]]
    base = yolo2b.StandardYolo2Generate()
    format = base.FindSameCellLocation(scalar, gt_box)
    pass


if __name__ == '__main__':
    if options.TestYolo2Build:
        __tes_yolo2_build()
        pass
    if options.TestFindSameCellLocation:
        __test___find_same_cell_location()
    pass

