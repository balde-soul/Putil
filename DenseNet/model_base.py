# coding = utf-8
import tensorflow as tf
from tensorflow.contrib import layers
import json


class DenseNet:
    def __init__(self, config_file):
        with open(config_file) as fp:
            self._Config = json.loads(fp.read())
            pass
        self._Placeholder = dict()
        self._Placeholder['training'] = tf.placeholder(dtype=tf.bool, name='training')
        pass

    def net_pro(self, feed):
        block_count = 0
        for block in self._Config['Arch']:
            name = ('block_' + str(block_count)) if (block['Name'] is None) else block['Name']
            channel = feed.get_shape().as_list()[-1]
            grow = block['Grow']
            with tf.name_scope(name):
                feed = tf.layers.batch_normalization(
                    feed,
                    training=self._Placeholder['training'],
                    gamma_initializer=tf.zeros_initializer,
                    beta_initializer=tf.random_normal_initializer,
                    moving_mean_initializer=tf.zeros_initializer,
                    moving_variance_initializer=tf.zeros_initializer,
                    beta_regularizer=layers.l1_regularizer,
                    gamma_regularizer=layers.l2_regularizer,
                    name='batch_normal'
                ) if block['BatchNormal'] is True else feed = feed
                with tf.name_scope('conv'):
                    feed = tf.layers.conv2d(
                        feed,
                        filters=channel + grow,
                        kernel_size=(block['Height'], block['Width']),
                        strides=(1, 1),
                        padding="SAME"
                    )
        pass



