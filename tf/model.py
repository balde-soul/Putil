# coding=utf-8
'''
_save_dir:
_weight_dir:
_summary_dir:
_train_summary_key:
_evaluate_summary_key:
_test_summary_key:
_training:
_step:
_train_op:
'''
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class Model(metaclass=ABCMeta):
    def __init__(self, graph_name):
        self._save_dir = None
        self._weight_fold_name = None
        self._summary_fold_name = None
        
        tf.GraphKeys.TRAIN = 'Train'
        self._train_summary_key = tf.GraphKeys.TRAIN
        tf.GraphKeys.EVALUATE = 'Evaluate'
        self._evaluate_summary_key = tf.GraphKeys.EVALUATE
        tf.GraphKeys.TEST = 'Test'
        self._test_summary_key = tf.GraphKeys.TEST

        with tf.variable_scope(graph_name):
            self._training = tf.placeholder(dtype=tf.bool, shape=[], name='training')
            self._step = tf.placeholder
        
        self._loss = None
        self._train_op = None
        pass