# coding = utf-8

import tensorflow as tf


def _get_operation(global_operation, name):
    for i in global_operation:
        if i.name == name:
            return i
        else:
            pass
        pass
    pass


def original_apply_moving(sess):
    sess = tf.Session()
    moving = sess.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
    global_operation = sess.graph.get_operations()
    map = dict()
    for i in moving:
        map[i.name] = _get_operation(global_operation, i.name.replace('/ExponentialMovingAverage', ''))
        pass
    return map

if __name__ == '__main__':

    print('------test original_apply_moving------')
    sess.close()
    tf.reset_default_graph()
    pass
