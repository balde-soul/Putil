# coding = utf-8
import tensorflow as tf


# todo: generate initialize method, coding
def initialize(method, dtype):
    """
    
    :param type: 
    :return: 
    """
    init = tf.zeros_initializer(dtype) if method == 'zero' else None
    init = tf.ones_initializer(dtype=dtype) if method == 'one' else None
