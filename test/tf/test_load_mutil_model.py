# coding=utf-8
import tensorflow as tf
import threading
import os
import os.path as osp
import sys


data_path = '../../data/tf'
if not osp.exists(data_path):
    sys.exit()
    pass

mutil_model_test_path = osp.join(data_path, 'test_mutil_model')
if not osp.exists(mutil_model_test_path):
    sys.exit()
    pass

one_path = osp.join(mutil_model_test_path, '1')
if not osp.exists(one_path):
    sys.exit()
    pass
two_path = osp.join(mutil_model_test_path, '2')
if not osp.exists(two_path):
    sys.exit()
    pass


class LoadModel(threading.Thread):
    def __init__(self, func, args=()):
        super(LoadModel, self).__init__()
        self._func = func
        self._args = args
        self._result = None
        pass

    def run(self):
        self._result = self._func(*self._args)
        pass

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self._result
        except Exception as ex:
            return None
        pass
    pass


def LoadOne():
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        saver = tf.train.import_meta_graph(osp.join(one_path, 'model.meta'))
    saver.restore(sess, osp.join(one_path, 'model'))
    target = sess.graph.get_tensor_by_name('default/add:0')
    source = sess.graph.get_tensor_by_name('default/input:0')
    print("model one load successfully")
    return sess, source, target
    pass


def Run(sess, source, target, feed, repeat):
    for i in range(0, repeat):
        print('{0}:{1}'.format(threading.get_ident(), sess.run(target, feed_dict={source: feed})))


def LoadTwo():
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        saver = tf.train.import_meta_graph(osp.join(two_path, 'model.meta'))
    saver.restore(sess, osp.join(two_path, 'model'))
    target = sess.graph.get_tensor_by_name('default/mul_1:0')
    source = sess.graph.get_tensor_by_name('default/input:0')
    print("model two load successfully")
    return sess, source, target
    pass


sess1, source1, target1 = LoadOne()
sess2, source2, target2 = LoadTwo()


one = LoadModel(Run, (sess1, source1, target1, 2.0, 100))
two = LoadModel(Run, (sess2, source2, target2, 2.0, 100))


one.start()
two.start()

one.join()
two.join()
