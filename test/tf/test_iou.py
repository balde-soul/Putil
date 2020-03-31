# coding=utf-8
import Putil.base.logger as plog
plog.set_internal_debug_log()

import tensorflow as tf
import Putil.tf.iou as piou
import numpy as np

tshape = [2, 32, 32, 2]
pre = tf.placeholder(dtype=tf.int32, shape=tshape)
gt = tf.placeholder(dtype=tf.int32, shape=tshape)

iou = piou.calc_semantic_iou(pre, gt, tf.float32)

session = tf.Session()

dpre = np.zeros(shape=tshape, dtype=np.int32)
print(session.run(iou, feed_dict={pre: dpre, gt: dpre}))

dpre = np.ones(shape=tshape, dtype=np.int32)
print(session.run(iou, feed_dict={pre: dpre, gt: dpre}))

dpre = np.zeros(shape=tshape, dtype=np.int32)
dpre[0, 7: 23, 7: 23, 0] = 1
dpre[0, 7: 15, 7: 23, 1] = 1
dgt = np.zeros(shape=tshape, dtype=np.int32)
dgt[0, 15: 30, 15: 30, 0] = 1
dgt[0, 0: 15, 0: 15, 1] = 1
print(session.run(iou, feed_dict={pre: dpre, gt: dgt}))