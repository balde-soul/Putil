# coding = utf-8

import tensorflow as tf
import Putil.tf.util as tfu
import Putil.np.util as npu
import logging
import sys
from colorama import Fore
import json
import numpy as np
import os
import tensorflow.contrib.layers as tf_layers


# ++: apply softmax cross dim(3)
def pixel_wise_softmax(output_map):
    with tf.name_scope('pixel_wise_softmax'):
        exp = tf.exp(output_map, name='exp')
        normalize = tf.reduce_sum(exp, axis=3, keepdims=True)
        return tf.div(exp, normalize, name='normalize')
    pass


# : activate layer
def Ac(output_map, param, **options):
    """
    :param output_map:
    :param param:{}, test/tf/test_layer/ac_config.json中提供了参考参数
    :return:
    """
    _name = options.pop('name', 'Ac')
    _method = param.get("type")
    print('->Ac')
    with tf.name_scope(_name):
        if _method == "ReLU":
            print(Fore.LIGHTGREEN_EX + '-->use ReLU')
            return tf.nn.relu(output_map, name="ReLU")
        elif _method == "PReLU":
            print(Fore.LIGHTGREEN_EX + '-->use PReLU')
            _type = tfu.tf_type(param.get("param_type")).Type
            _alpha = param.get("alpha")
            _alpha_trainable = param.get('alpha_trainable', False)
            if _alpha_trainable:
                print(Fore.LIGHTGREEN_EX + '--->alpha trainable')
                with tf.name_scope("alpha_trainable"):
                    alpha = tf.Variable(_alpha, trainable=True, dtype=_type)
                    p = tf.nn.relu(output_map, name='positive')
                    n = tf.subtract(output_map, p, name='sub_p_get_n')
                    n_mul = tf.multiply(n, alpha, name='n_mul')
                    return tf.add(n_mul, p, name='PReLU')
            else:
                print(Fore.LIGHTGREEN_EX + '---->alpha untrainable')
                with tf.name_scope('alpha_un_trainable'):
                    return tf.nn.leaky_relu(output_map, _alpha, name="PReLU")
                pass
            pass
        else:
            logging.ERROR(Fore.RED('unsupported method: {0}'.format(
               _method)))
            sys.exit()
            pass
        pass
    pass


# todo: get the initializer
def Initializer(type, param, **options):
    if type == 'mrsa':
        _factor = param.get('factor', 2.0)
        _mode = param.get('mode', 'FAN_IN')
        _uniform = param.get('uniform', False)
        _seed = param.get('seed', None)
        try:
            _type = param.get('param_type')
        except KeyError as e:
            print(Fore.RED + "initializer param: {0} lack key: {1}".format(param, str(e)))
            sys.exit()
            pass
        tf_layers.variance_scaling_initializer(_factor, _mode, _uniform, _seed, _type)
    else:
        print(Fore.RED + '{0} initializer method is not supported!'.format(type))
        sys.exit()
        pass
    pass


# todo: conv2d
def Conv(output_map, param, name, **options):
    _type = param.get('param_type')
    _padding = param.get('padding')
    _feature_map_amount = param['feature_map_amount']
    _height = param['weight']['height']
    _width = param['weight']['width']
    _stride_h = param['weight']['stride_h']
    _stride_w = param['weight']['stride_w']
    _initializer_w = param['weight']['initializer']
    _initializer_w_param = param['weight']['initializer_param']

    _bias_on = param['bias']['on']
    with tf.variable_scope(name, **options):
        weight_initializer = Initializer(_initializer_w, _initializer_w_param)

    pass


def __test_ac(param):
    print(Fore.GREEN + '-------test ac--------')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.reset_default_graph()

    if param['type'] == 'ReLU':
        output_map = tf.constant([-1, 1, -2, 3])
        _ac = Ac(output_map, param)
        sess = tf.Session()
        ac = sess.run(_ac)
        print(Fore.GREEN + '>>------test ReLU------<<')
        assert (ac == np.array([0, 1, 0, 3],
                               dtype=tfu.tf_type.to_np(output_map.dtype))).all(), print(
            Fore.RED + 'ReLU failed except: [0, 1, 0, 3] , but get {result}'.format(
                result=_ac))
        pass
    elif param['type'] == 'PReLU':
        output_map = tf.constant([-1, 1, -2, 3], tfu.tf_type(param['param_type']).Type)
        _ac = Ac(output_map, param)
        sess = tf.Session()
        ac = sess.run(_ac)
        print(Fore.GREEN + '>>------test PReLU------<<')
        expect_result = np.array([-1 * param['alpha'], 1, -2 * param['alpha'], 3],
                                 dtype=npu.np_type(param['param_type']).Type)
        try:
            assert (ac == expect_result).all()
        except AssertionError:
            print(Fore.RED + ('PReLU failed except: {0}, but get {1}'.format(
                expect_result, ac)))
        pass
    print(Fore.GREEN + '***>>{type} test pass '.format(
        type=param['type']))
    pass


def __Test(options):
    # : test ac
    if options.TestAc:
        with open(options.AcConfig, 'r') as fp:
            params = json.loads(fp.read())
            pass
        for param in params.values():
            __test_ac(param)
            pass
        pass
    else:
        pass
    pass


if __name__ == '__main__':
    from optparse import OptionParser
    usage = "show something usefull -- for example: how to use this program"
    parser = OptionParser(usage)
    parser.add_option(
        '--test_ac',
        action='store_true',
        default=False,
        dest='TestAc',
        help='while you want to test the Ac function set this flag\n'
             'default: False'
    )
    parser.add_option(
        '--ac_config',
        action='store',
        default='../test/tf/test_layer/ac_config.json',
        dest='AcConfig',
        help='while set test_ac , specify the config json file\n'
             'default: ../test/tf/test_layer/ac_config.json'
    )

    (options, args) = parser.parse_args()

    __Test(options)


