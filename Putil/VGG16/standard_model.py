import tensorflow as tf


class Model:
    def __init__(self):
        pass

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
            conv = tf.layers.conv2d(
        with tf.variable_scope(name):
                bottom,
                filters=grow,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
                kernel_regularizer=layers.l2_regularizer(regularize_weight),
                bias_initializer=tf.zeros_initializer(dtype=param_dtype),
                bias_regularizer=layers.l2_regularizer(regularize_weight),
                name='conv'
            )

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def build(self, feed, train=False):
        self.relu1_1 = self._conv_layer(feed, "conv1_1")
        self.relu1_2 = self._conv_layer(self.relu1_1, "conv1_2")
        self.pool1 = self._max_pool(self.relu1_2, 'pool1')

        self.relu2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.relu2_2 = self._conv_layer(self.relu2_1, "conv2_2")
        self.pool2 = self._max_pool(self.relu2_2, 'pool2')

        self.relu3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.relu3_2 = self._conv_layer(self.relu3_1, "conv3_2")
        self.relu3_3 = self._conv_layer(self.relu3_2, "conv3_3")
        self.pool3 = self._max_pool(self.relu3_3, 'pool3')

        self.relu4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.relu4_2 = self._conv_layer(self.relu4_1, "conv4_2")
        self.relu4_3 = self._conv_layer(self.relu4_2, "conv4_3")
        self.pool4 = self._max_pool(self.relu4_3, 'pool4')

        self.relu5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.relu5_2 = self._conv_layer(self.relu5_1, "conv5_2")
        self.relu5_3 = self._conv_layer(self.relu5_2, "conv5_3")
        self.pool5 = self._max_pool(self.relu5_3, 'pool5')
        pass
    pass


if __name__ == '__main__':
    vgg16 = Model()
    feed = tf.placeholder(tf.float32, [10, 224, 224, 3], name='feed')
    vgg16.build(feed)
    pass
