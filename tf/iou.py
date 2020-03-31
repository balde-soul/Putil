# coding=utf-8
import tensorflow as tf


def calc_semantic_iou(pre, gt, dtype):
    '''
    pre: the onehot prediction [batch, height, width, class_amount]
    gt: the onehot ground truth [batch, height, width, class_amount]
    '''
    tp = tf.multiply(pre, gt)
    fp = tf.multiply(pre, tf.subtract(1, gt))
    fn = tf.multiply(tf.subtract(1, pre), gt)
    tn = tf.multiply(tf.subtract(1, pre), tf.subtract(1, gt))
    tps = tf.cast(tf.reduce_sum(tp, axis=[1, 2]), dtype=dtype)
    fps = tf.cast(tf.reduce_sum(fp, axis=[1, 2]), dtype=dtype)
    fns = tf.cast(tf.reduce_sum(fn, axis=[1, 2]), dtype=dtype)
    iou = tps / (tps + fps + fns + 1e-32)
    return iou
    pass