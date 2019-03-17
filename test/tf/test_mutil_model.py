# coding=utf-8
import tensorflow as tf
import tensorflow.contrib as contrib
import threading
import os
import os.path as osp
# write model

one_can_run = False
two_can_run = False
two_can_nd_run = False
one_locker = threading.Lock()
two_locker = threading.Lock()
two_nd_locker = threading.Lock()
cond_one = threading.Condition(one_locker)
cond_two = threading.Condition(two_locker)
cond_two_nd = threading.Condition(two_nd_locker)


def one_flag_get():
    return one_can_run


def two_flag_get():
    return two_can_run


def two_nd_flag_get():
    return two_can_nd_run


class Model(threading.Thread):
    def __init__(self, func, args=()):
        super(Model, self).__init__()
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


one_path = '../../data/tf/test_mutil_model/1'
two_path = '../../data/tf/test_mutil_model/2'
if not osp.exists(one_path):
    os.mkdir(one_path)
    pass

if not osp.exists(two_path):
    os.mkdir(two_path)
    pass


def build_one():
    global two_can_run
    global two_can_nd_run
    graph1 = tf.Graph()
    sess1 = tf.Session(graph=graph1)

    print(sess1.graph_def)
    print('------------------------')
    with graph1.as_default():
        with tf.name_scope('default'):
            i1 = tf.placeholder(dtype=tf.float32, name='input')
            var1 = tf.get_variable(name='var1', shape=[], dtype=tf.float32)
            # var1 = 2.5
            m_1 = tf.multiply(i1, var1, name='mul')
            two_can_run = True
            cond_two.acquire()
            cond_two.notify_all()
            cond_two.release()
            cond_one.acquire()
            print('wait for on_can_run')
            cond_one.wait_for(one_flag_get)
            print('get one_can_run')
            var2 = tf.get_variable(name='var2', shape=[], dtype=tf.float32)
            # var2 = 1.9
            a_1 = tf.add(m_1, var2, name='add')
        sess1.run(tf.global_variables_initializer())
        saver1 = tf.train.Saver(tf.global_variables())
        tf.summary.FileWriter(one_path, graph=tf.get_default_graph())
        # saver1 = []
        pass
    two_can_nd_run = True
    cond_two_nd.acquire()
    cond_two_nd.notify_all()
    cond_two_nd.release()
    return sess1, saver1


def build_two():
    global one_can_run
    graph2 = tf.Graph()
    sess2 = tf.Session(graph=graph2)

    print(sess2.graph_def)
    print('------------------------')

    cond_two.acquire()
    print('wait for two_can_run')
    cond_two.wait_for(two_flag_get)
    print('get two_can_run')
    with graph2.as_default():
        with tf.name_scope('default'):
            i1_2 = tf.placeholder(dtype=tf.float32, name='input')
            var1_2 = tf.get_variable(name='var1', shape=[], dtype=tf.float32)
            # var1_2 = 2.0
            m_1_2 = tf.multiply(i1_2, var1_2, name='mul')
            var2_2 = tf.get_variable(name='var2', shape=[], dtype=tf.float32)
            one_can_run = True
            cond_one.acquire()
            cond_one.notify_all()
            cond_one.release()
            cond_two_nd.acquire()
            print('wait for two_can_nd_run')
            cond_two_nd.wait_for(two_nd_flag_get)
            print('get two_can_nd_run')
            # var2_2 = 1.5
            m_2_2 = tf.multiply(m_1_2, var2_2, name='mul')

        sess2.run(tf.global_variables_initializer())
        saver2 = tf.train.Saver(tf.global_variables())
        tf.summary.FileWriter(two_path, graph=tf.get_default_graph())
        # saver2 = []
        pass
    return sess2, saver2


thread1 = Model(build_one)
thread2 = Model(build_two)

thread1.start()
thread2.start()

print(thread1.get_result()[0].graph_def)
print('----------------------------')
print(thread2.get_result()[0].graph_def)
print('----------------------------')

thread1.get_result()[1].save(thread1.get_result()[0], osp.join(one_path, 'model'))
thread2.get_result()[1].save(thread2.get_result()[0], osp.join(two_path, 'model'))


