# coding = utf-8
import tensorflow as tf
from tensorflow.contrib import layers
import json
import Putil.tf.util as tfu
from colorama import Fore
import sys
import Putil.tf.layers as p_layers


class DeDenseNetProvide:
    def __init__(self):
        self._transition = list()
        self._block_layer = list()
        self._block_amount = -1
        pass

    def push_transition(self, transition_output):
        self._transition.append(transition_output)
        pass

    def push_block(self):
        self._block_amount += 1
        self._block_layer.append(list())
        pass

    def push_block_layer(self, block_layer_output):
        self._block_layer[self._block_amount].append(block_layer_output)
        pass

    @property
    def Transition(self):
        return self._transition

    @property
    def BlockAmount(self):
        return self._block_amount

    @property
    def BlockLayer(self):
        return self._block_layer

    pass


def DeDenseNetFromConfigFile(
        output_map,
        training,
        config_file,
        block_name_flag='',
        dense_net_provide=None
):
    """

    :param output_map: feature from other net
    :param training:
    :param config_file:
    :param block_name_flag: the prefix for the name
    :param dense_net_provide: the collector
    :return:
    """
    options = {'block_name_flag': block_name_flag,
               'de_dense_net_provide': dense_net_provide}
    with open(config_file, 'r') as fp:
        params = json.loads(fp.read())['DeDenseNet']
        pass
    output_map = DeDenseNetFromParamDict(
        output_map,
        training,
        params,
        **options
    )
    return output_map
    pass


def DeDenseNetFromParamDict(
        output_map,
        training,
        params,
        **options
):
    """

    :param output_map:
    :param training:
    :param params:
        {
  "DeDenseNet":[
    {
      "param_dtype": 0.32,

      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],

      "t_kernel": [3, 3],
      "t_stride": [2, 2],
      "compress_rate": 0.3,

      "layer_param":{
        "batch_normal": true,
        "activate":{
          "type": "ReLU"
        }
      },

      "transition_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        },
        "dropout_rate": 0.1
      }
    },
    {
      "param_dtype": 0.32,

      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],

      "t_kernel": [3, 3],
      "t_stride": [2, 2],
      "compress_rate": 0.3,

      "layer_param":{
        "batch_normal": true,
        "activate":{
          "type": "ReLU"
        }
      },

      "transition_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        },
        "dropout_rate": 0.1
      }
    },
    {
      "param_dtype": 0.32,

      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],

      "t_kernel": [3, 3],
      "t_stride": [2, 2],
      "compress_rate": 0.3,

      "layer_param":{
        "batch_normal": true,
        "activate":{
          "type": "ReLU"
        }
      },

      "transition_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        },
        "dropout_rate": 0.1
      }
    },
    {
      "param_dtype": 0.32,

      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],

      "t_kernel": [3, 3],
      "t_stride": [2, 2],
      "compress_rate": 0.3,

      "layer_param":{
        "batch_normal": true,
        "activate":{
          "type": "ReLU"
        }
      },

      "transition_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        },
        "dropout_rate": 0.1
      }
    }
  ]
}
    :param options:
        block_name_flag: the prefix for name
        de_dense_net_provide: the collector
    :return:
    """
    block_name_flag = options.pop('block_name_flag', '')
    dense_net_provide = options.pop('de_dense_net_provide', None)
    block_name = 0
    for param in params:
        param_dtype = param.get('param_dtype')
        grows = param.get('grows')
        name = str(block_name)
        regularize_weight = param.get('regularize_weight')
        kernels = param.get('kernels')
        t_kernel = param.get('t_kernel')
        t_stride = param.get('t_stride')
        compress_rate = param.get('compress_rate')
        layer_param = param.get('layer_param')
        layer_param['training'] = training
        transition_param = param.get('transition_param')
        transition_param['training'] = training
        output_map = DeDenseNetBlock(
            output_map,
            tfu.tf_type(param_dtype).Type,
            grows,
            '{0}{1}'.format(block_name_flag, name),
            regularize_weight,
            kernels,
            t_kernel,
            t_stride,
            compress_rate,
            layer_param,
            transition_param,
            dense_net_provide
        )
        block_name += 1
        pass
    return output_map
    pass


def DeDenseNetBlock(
        output_map,
        param_dtype,
        grows,
        name,
        regularize_weight,
        kernels,
        t_kernel,
        t_stride,
        compress_rate,
        layer_param,
        transition_param,
        dense_net_provide=None
):
    """

    :param output_map:
    :param param_dtype:
    :param grows:
    :param name:
    :param regularize_weight:
    :param kernels:
    :param t_kernel:
    :param t_stride:
    :param compress_rate:
    :param layer_param:
    :param transition_param:
    :param dense_net_provide:
    :return:
    """
    # transition
    print(Fore.LIGHTGREEN_EX + '>>>>>>>>>>XXXXXXXXXX block transition XXXXXXXXXX<<<<<<<<<<')
    print(Fore.LIGHTGREEN_EX + '')
    output_map = DeDenseNetBlockTransition(
        output_map,
        param_dtype,
        '{0}_transition'.format(name),
        regularize_weight,
        t_kernel,
        t_stride,
        compress_rate,
        **transition_param
    )
    if dense_net_provide is not None:
        dense_net_provide.push_transition(output_map)
        pass
    # block layer
    output_map = DeDenseNetBlockLayers(
        output_map,
        param_dtype,
        grows,
        name,
        regularize_weight,
        kernels,
        layer_param,
        dense_net_provide=dense_net_provide
    )
    # block_layer_count = 0
    # print(Fore.LIGHTGREEN_EX + 'XXXXXXXXXXXXXXXXXXXX block {0} XXXXXXXXXXXXXXXXX'.format(name))
    # print(Fore.LIGHTGREEN_EX + '>>>>>>>>>>XXXXXXXXXX block layer XXXXXXXXXX<<<<<<<<<<')
    # if dense_net_provide is not None:
    #     dense_net_provide.push_block()
    #     pass
    # for _shape in zip(grows, kernels):
    #     output_map = DeDenseNetBlockLayer(
    #         output_map,
    #         param_dtype,
    #         regularize_weight,
    #         '{0}_{1}'.format(name, block_layer_count),
    #         _shape[0],
    #         _shape[1],
    #         **layer_param
    #     )
    #     if dense_net_provide is not None:
    #         dense_net_provide.push_block_layer(output_map)
    #         pass
    #     block_layer_count += 1
    #     pass
    return output_map
    pass


def DeDenseNetBlockLayers(
        output_map,
        param_dtype,
        grows,
        name,
        regularize_weight,
        kernels,
        layer_param,
        **options
):
    dense_net_provide = options.pop('dense_net_provide', None)
    block_layer_count = 0
    print(Fore.LIGHTGREEN_EX + 'XXXXXXXXXXXXXXXXXXXX block {0} XXXXXXXXXXXXXXXXX'.format(name))
    print(Fore.LIGHTGREEN_EX + '>>>>>>>>>>XXXXXXXXXX block layer XXXXXXXXXX<<<<<<<<<<')
    if dense_net_provide is not None:
        dense_net_provide.push_block()
        pass
    for _shape in zip(grows, kernels):
        output_map = DeDenseNetBlockLayer(
            output_map,
            param_dtype,
            regularize_weight,
            '{0}_{1}'.format(name, block_layer_count),
            _shape[0],
            _shape[1],
            **layer_param
        )
        if dense_net_provide is not None:
            dense_net_provide.push_block_layer(output_map)
            pass
        block_layer_count += 1
        pass
    return output_map
    pass

def DeDenseNetBlockLayer(
        output_map,
        param_dtype,
        regularize_weight,
        name,
        grow,
        kernel,
        **options
):
    """

    :param output_map: tf.operation}feature from other net
    :param param_dtype: {tf.dtype}param type special
    :param regularize_weight: {float}weight regularize_weight
    :param name: {str}transition name
    :param grow: {int} specify the amount of output feature int the convolution layer
    :param kernel: {list} specify the weight_size used in convolution layer,[height, width]
    :param options:
        batch_normal: {bool}while you want to use batch normal set this
        training: {tf.placeholder}while use batch normal , this should be specified
        activate_param: {dict}from Putil.tf.layers.Ac config dict, default is ReLU
    :return:
    """
    feed_cord = output_map
    print(Fore.LIGHTGREEN_EX + '******************block:{0}***************'.format(name))
    with tf.variable_scope(name) as block_layer_scope:
        # for batch normal
        print(Fore.LIGHTGREEN_EX + '-----------------bn-----------------')
        batch_normal = options.pop('batch_normal', True)
        training = options.pop('training') if batch_normal is True else None
        # : print Arch Info
        print(Fore.LIGHTGREEN_EX + 'param type: {0}'.format(param_dtype.name))
        if batch_normal:
            print(Fore.LIGHTGREEN_EX + 'use batch normal')
            assert training.dtype.name == 'bool', \
                print(Fore.LIGHTRED_EX + 'training should be bool tf.placeholder')
        else:
            print(Fore.LIGHTGREEN_EX + 'do not use batch normal')
            pass
        # options: batch normal
        output_map = tf.layers.batch_normalization(
            output_map,
            training=training,
            gamma_initializer=tf.zeros_initializer(param_dtype),
            beta_initializer=tf.zeros_initializer(param_dtype),
            moving_mean_initializer=tf.zeros_initializer(dtype=param_dtype),
            moving_variance_initializer=tf.zeros_initializer(dtype=param_dtype),
            beta_regularizer=layers.l2_regularizer(regularize_weight),
            gamma_regularizer=layers.l2_regularizer(regularize_weight),
            name='batch_normal'
        ) if batch_normal is True else output_map
        # ac
        # for activate
        print(Fore.LIGHTGREEN_EX + '-----------------Ac--------------')
        ac_param = options.pop('activate_param', {'type': 'ReLU'})
        output_map = p_layers.Ac(output_map, ac_param)
        # conv
        # for convolution
        print(Fore.LIGHTGREEN_EX + '-----------------Conv--------------')
        print(Fore.LIGHTGREEN_EX + 'grow: {0}'.format(grow))
        print(Fore.LIGHTGREEN_EX + 'kernel size: {0}'.format(kernel))
        output_map = tf.layers.conv2d(
            output_map,
            filters=grow,
            kernel_size=kernel,
            strides=(1, 1),
            padding="same",
            kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
            kernel_regularizer=layers.l2_regularizer(regularize_weight),
            bias_initializer=tf.zeros_initializer(dtype=param_dtype),
            bias_regularizer=layers.l2_regularizer(regularize_weight),
            name='conv'
        )
        # concat
        print(Fore.LIGHTGREEN_EX + '-------------------concat------------------')
        output_map = tf.concat([output_map, feed_cord], axis=-1, name='block_element')
        pass
    return output_map
    pass


# todo: use output_map
def DeDenseNetBlockTransition(
        output_map,
        param_dtype,
        name,
        regularize_weight,
        kernel,
        stride,
        compress_rate,
        **options
):
    """

    :param output_map:
    :param param_dtype: tf.dtype
    :param name: name for named
    :param regularize_weight:
    :param kernel: {list} for de conv , [height, width]
    :param stride: {lits} for de conv which decide the rate de conv , [height, width]
    :param compress_rate: which decide the generating feature map amount, float
            while compress_rate < 1.0 Generate_feature_map_amount < Input_feature_map
    :param options:
            batch_normal: {bool}while you want to use batch normal set this
            training: {tf.placeholder}while use batch normal , this should be specified
            activate_param: {dict}from Putil.tf.layers.Ac config dict, default is ReLU
            dropout_rate: {float}{while you want to apply dropout , you should specify this
    :return
    """
    print(Fore.LIGHTGREEN_EX + '******************block_{0}_transition'.format(name))
    with tf.variable_scope(name) as block_transition_scope:
        # for batch normal
        print(Fore.LIGHTGREEN_EX + '-----------------bn-----------------')
        batch_normal = options.pop('batch_normal', True)
        training = options.pop('training') if batch_normal is True else None
        # : print Arch Info
        print(Fore.LIGHTGREEN_EX + 'param type: {0}'.format(param_dtype.name))
        if batch_normal:
            print(Fore.LIGHTGREEN_EX + 'use batch normal')
            assert training.dtype.name == 'bool', \
                print(Fore.LIGHTRED_EX + 'training should be bool tf.placeholder')
        else:
            print(Fore.LIGHTGREEN_EX + 'do not use batch normal')
            pass
        # options: batch normal
        output_map = tf.layers.batch_normalization(
            output_map,
            training=training,
            gamma_initializer=tf.zeros_initializer(param_dtype),
            beta_initializer=tf.zeros_initializer(param_dtype),
            moving_mean_initializer=tf.zeros_initializer(dtype=param_dtype),
            moving_variance_initializer=tf.zeros_initializer(dtype=param_dtype),
            beta_regularizer=layers.l2_regularizer(regularize_weight),
            gamma_regularizer=layers.l2_regularizer(regularize_weight),
            name='batch_normal'
        ) if batch_normal is True else output_map
        # ac
        # for activate
        print(Fore.LIGHTGREEN_EX + '-----------------Ac--------------')
        ac_param = options.pop('activate_param', {'type': 'ReLU'})
        output_map = p_layers.Ac(output_map, ac_param)
        # Convolution
        print(Fore.LIGHTGREEN_EX + '------------------DeConv---------------')
        channel = int(output_map.get_shape().as_list()[-1] * (1 - compress_rate)) if compress_rate is not None \
            else output_map.get_shape().as_list()[-1]
        print(Fore.LIGHTGREEN_EX + 'compress rate: {0}\noutput feature map amount: {1}'.format(
            compress_rate, channel
        ))
        output_map = tf.layers.conv2d_transpose(
            output_map,
            filters=channel,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
            kernel_regularizer=layers.l2_regularizer(regularize_weight),
            bias_initializer=tf.zeros_initializer(dtype=param_dtype),
            bias_regularizer=layers.l2_regularizer(regularize_weight),
            name='conv-transpose'
        )
        # Optional dropout
        print(Fore.LIGHTGREEN_EX + '------------------dropout---------------')
        dropout_rate = options.pop('dropout_rate', None)
        output_map = tf.layers.dropout(output_map, dropout_rate, training=training, name='dropout') \
            if dropout_rate is not None else output_map
        pass
    return output_map
    pass


class DenseNetProvide:
    def __init__(self):
        self._transition = list()
        self._block_layer = list()
        self._block_amount = -1
        pass

    def push_transition(self, transition_output):
        self._transition.append(transition_output)
        pass

    def push_block(self):
        self._block_amount += 1
        self._block_layer.append(list())
        pass

    def push_block_layer(self, block_layer_output):
        self._block_layer[self._block_amount].append(block_layer_output)
        pass

    @property
    def Transition(self):
        return self._transition

    @property
    def BlockAmount(self):
        return self._block_amount

    @property
    def BlockLayer(self):
        return self._block_layer

    pass


def DenseNetFromConfigFile(
        output_map,
        training,
        config_file,
        block_name_flag='',
        dense_net_provide=None
):
    """
    generate DenseNet from Config file
    :param output_map: feature map from other net
    :param training: placeholder for training or not
    :param config_file: path for the config file
    :param block_name_flag:
    :param dense_net_provide:
    :return:
    """
    options={'block_name_flag': block_name_flag,
             'dense_net_provide': dense_net_provide}
    with open(config_file, 'r') as fp:
        params = json.loads(fp.read())['DenseNet']
        pass
    output_map = DenseNetFromParamDict(
        output_map,
        training,
        params,
        **options
    )
    return output_map
    pass


def DenseNetFromParamDict(
        output_map,
        training,
        params,
        **options
):
    """

    :param output_map: feature from other net
    :param training: placeholder for training or not
    :param params: param list such as:
    [
    {
      "param_dtype": 0.32,
      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],
      "pool_kernel": [2, 2],
      "pool_stride": [2, 2],
      "pool_type": "max",
      "layer_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        }
      },
      "transition_param":{
        "batch_normal": true,
        "activate_param": {
          "type": "ReLU"
        },
        "compress_rate": null,
        "dropout_rate": 0.1
      }
    },
    {
      "param_dtype": 0.32,
      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],
      "pool_kernel": [2, 2],
      "pool_stride": [2, 2],
      "pool_type": "max",
      "layer_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        }
      },
      "transition_param":{
        "batch_normal": true,
        "activate_param": {
          "type": "ReLU"
        },
        "compress_rate": null,
        "dropout_rate": 0.1
      }
    },
    {
      "param_dtype": 0.32,
      "grows": [3, 3, 3],
      "regularize_weight": 0.0001,
      "kernels": [[3, 3], [3, 3], [3, 3]],
      "pool_kernel": [2, 2],
      "pool_stride": [2, 2],
      "pool_type": "max",
      "layer_param":{
        "batch_normal": true,
        "activate_param":{
          "type": "ReLU"
        }
      },
      "transition_param":{
        "batch_normal": true,
        "activate_param": {
          "type": "ReLU"
        },
        "compress_rate": null,
        "dropout_rate": 0.1
      }
    }
  ]
    :return:
    """
    block_name_flag = options.pop('block_name_flag', '')
    dense_net_provide = options.pop('dense_net_provide', None)
    block_name = 0
    for param in params:
        param_dtype = param.get('param_dtype')
        grows = param.get('grows')
        name = str(block_name)
        regularize_weight = param.get('regularize_weight')
        kernels = param.get('kernels')
        pool_kernel = param.get('pool_kernel')
        pool_stride = param.get('pool_stride')
        pool_type = param.get('pool_type')
        layer_param = param.get('layer_param')
        layer_param['training'] = training
        transition_param = param.get('transition_param')
        transition_param['training'] = training
        output_map = DenseNetBlock(
            output_map,
            tfu.tf_type(param_dtype).Type,
            grows,
            '{0}{1}'.format(block_name_flag, name),
            regularize_weight,
            kernels,
            pool_kernel,
            pool_stride,
            pool_type,
            layer_param,
            transition_param,
            dense_net_provide
        )
        block_name += 1
        pass
    return output_map
    pass


def DenseNetBlock(
        output_map,
        param_dtype,
        grows,
        name,
        regularize_weight,
        kernels,
        pool_kernel,
        pool_stride,
        pool_type,
        layer_param,
        transition_param,
        dense_net_provide=None
):
    """

    :param output_map:
    :param param_dtype:
    :param grows:
    :param name:
    :param regularize_weight:
    :param kernels:
    :param pool_kernel:
    :param pool_stride:
    :param layer_param:
    :param transition_param:
    :return:
    """
    block_layer_count = 0
    print(Fore.LIGHTGREEN_EX + 'XXXXXXXXXXXXXXXXXXXX block {0} XXXXXXXXXXXXXXXXX'.format(name))
    print(Fore.LIGHTGREEN_EX + '>>>>>>>>>>XXXXXXXXXX block layer XXXXXXXXXX<<<<<<<<<<')
    if dense_net_provide is not None:
        dense_net_provide.push_block()
        pass
    for _shape in zip(grows, kernels):
        output_map = DenseNetBlockLayer(
            output_map,
            param_dtype,
            regularize_weight,
            '{0}_{1}'.format(name, block_layer_count),
            _shape[0],
            _shape[1],
            **layer_param
        )
        if dense_net_provide is not None:
            dense_net_provide.push_block_layer(output_map)
            pass
        block_layer_count += 1
        pass
    print(Fore.LIGHTGREEN_EX + '>>>>>>>>>>XXXXXXXXXX block transition XXXXXXXXXX<<<<<<<<<<')
    print(Fore.LIGHTGREEN_EX + '')
    output_map = DenseNetBlockTransition(
        output_map,
        param_dtype,
        '{0}_transition'.format(name),
        regularize_weight,
        pool_kernel,
        pool_stride,
        pool_type,
        **transition_param
    )
    if dense_net_provide is not None:
        dense_net_provide.push_transition(output_map)
    return output_map
    pass


def DenseNetBlockLayer(
        output_map,
        param_dtype,
        regularize_weight,
        name,
        grow,
        kernel,
        **options):
    """

    :param output_map: tf.operation}feature from other net
    :param param_dtype: {tf.dtype}param type special
    :param name: {str}transition name
    :param regularize_weight: {float}weight regularize_weight
    :param grow: {int} specify the amount of output feature int the convolution layer
    :param kernel: {list} specify the weight_size used in convolution layer,[height, width]
    :param options:
        batch_normal: {bool}while you want to use batch normal set this
        training: {tf.placeholder}while use batch normal , this should be specified
        activate_param: {dict}from Putil.tf.layers.Ac config dict, default is ReLU
    :return:
    """
    feed_cord = output_map
    print(Fore.LIGHTGREEN_EX + '******************block:{0}***************'.format(name))
    with tf.variable_scope(name) as block_layer_scope:
        # for batch normal
        print(Fore.LIGHTGREEN_EX + '-----------------bn-----------------')
        batch_normal = options.pop('batch_normal', True)
        training = options.pop('training') if batch_normal is True else None
        # : print Arch Info
        print(Fore.LIGHTGREEN_EX + 'param type: {0}'.format(param_dtype.name))
        if batch_normal:
            print(Fore.LIGHTGREEN_EX + 'use batch normal')
            assert training.dtype.name == 'bool', \
                print(Fore.LIGHTRED_EX + 'training should be bool tf.placeholder')
        else:
            print(Fore.LIGHTGREEN_EX + 'do not use batch normal')
            pass
        # options: batch normal
        output_map = tf.layers.batch_normalization(
            output_map,
            training=training,
            gamma_initializer=tf.zeros_initializer(param_dtype),
            beta_initializer=tf.zeros_initializer(param_dtype),
            moving_mean_initializer=tf.zeros_initializer(dtype=param_dtype),
            moving_variance_initializer=tf.zeros_initializer(dtype=param_dtype),
            beta_regularizer=layers.l2_regularizer(regularize_weight),
            gamma_regularizer=layers.l2_regularizer(regularize_weight),
            name='batch_normal'
        ) if batch_normal is True else output_map
        # ac
        # for activate
        print(Fore.LIGHTGREEN_EX + '-----------------Ac--------------')
        ac_param = options.pop('activate_param', {'type': 'ReLU'})
        output_map = p_layers.Ac(output_map, ac_param)

        # conv
        # for convolution
        print(Fore.LIGHTGREEN_EX + '-----------------Conv--------------')
        print(Fore.LIGHTGREEN_EX + 'grow: {0}'.format(grow))
        print(Fore.LIGHTGREEN_EX + 'kernel size: {0}'.format(kernel))
        output_map = tf.layers.conv2d(
            output_map,
            filters=grow,
            kernel_size=kernel,
            strides=(1, 1),
            padding="same",
            kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
            kernel_regularizer=layers.l2_regularizer(regularize_weight),
            bias_initializer=tf.zeros_initializer(dtype=param_dtype),
            bias_regularizer=layers.l2_regularizer(regularize_weight),
            name='conv'
        )

        #concat
        print(Fore.LIGHTGREEN_EX + '-------------------concat------------------')
        output_map = tf.concat([output_map, feed_cord], axis=-1, name='block_element')
        pass
    return output_map

def DenseNetBlockLayers(
        output_map,
        param_dtype,
        grows,
        name,
        regularize_weight,
        kernels,
        layer_param,
        **options
):
    dense_net_provide = options.pop('dense_net_provide', None)
    block_layer_count = 0
    print(Fore.LIGHTGREEN_EX + 'XXXXXXXXXXXXXXXXXXXX block {0} XXXXXXXXXXXXXXXXX'.format(name))
    print(Fore.LIGHTGREEN_EX + '>>>>>>>>>>XXXXXXXXXX block layer XXXXXXXXXX<<<<<<<<<<')
    if dense_net_provide is not None:
        dense_net_provide.push_block()
        pass
    for _shape in zip(grows, kernels):
        output_map = DenseNetBlockLayer(
            output_map,
            param_dtype,
            regularize_weight,
            '{0}_{1}'.format(name, block_layer_count),
            _shape[0],
            _shape[1],
            **layer_param
        )
        if dense_net_provide is not None:
            dense_net_provide.push_block_layer(output_map)
            pass
        block_layer_count += 1
        pass
    return output_map
    pass


def DenseNetBlockTransition(
        output_map,
        param_dtype,
        name,
        regularize_weight,
        pool_kernel,
        pool_stride,
        pool_type,
        **options
):
    """

    :param output_map: {tf.operation}feature from other net
    :param param_dtype: {tf.dtype}param type special
    :param name: {str}transition name
    :param regularize_weight: {float}weight regularize_weight
    :param pool_kernel: {list}pool kernel size [height, width]
    :param pool_stride: {list}pool stride [vertical, horizontal]
    :param options:
        batch_normal: {bool}while you want to use batch normal set this
        training: {tf.placeholder}while use batch normal , this should be specified
        activate_param: {dict}from Putil.tf.layers.Ac config dict, default is ReLU
        compress_rate: {float}while you wan to compress feature map in transition ,you should specify this
                        while compress_rate < 1.0 Generate_feature_map_amount < Input_feature_map
        dropout_rate: {float}{while you want to apply dropout , you should specify this
    :return:
    """
    print(Fore.LIGHTGREEN_EX + '******************block_{0}_transition****************'.format(name))
    with tf.variable_scope(name) as block_transition_scope:
        # for batch normal
        print(Fore.LIGHTGREEN_EX + '-----------------bn-----------------')
        batch_normal = options.pop('batch_normal', True)
        training = options.pop('training') if batch_normal is True else None
        # : print Arch Info
        print(Fore.LIGHTGREEN_EX + 'param type: {0}'.format(param_dtype.name))
        if batch_normal:
            print(Fore.LIGHTGREEN_EX + 'use batch normal')
            assert training.dtype.name == 'bool', \
                print(Fore.LIGHTRED_EX + 'training should be bool tf.placeholder')
        else:
            print(Fore.LIGHTGREEN_EX + 'do not use batch normal')
            pass
        # options: batch normal
        output_map = tf.layers.batch_normalization(
            output_map,
            training=training,
            gamma_initializer=tf.zeros_initializer(param_dtype),
            beta_initializer=tf.zeros_initializer(param_dtype),
            moving_mean_initializer=tf.zeros_initializer(dtype=param_dtype),
            moving_variance_initializer=tf.zeros_initializer(dtype=param_dtype),
            beta_regularizer=layers.l2_regularizer(regularize_weight),
            gamma_regularizer=layers.l2_regularizer(regularize_weight),
            name='batch_normal'
        ) if batch_normal is True else output_map
        # ac
        # for activate
        print(Fore.LIGHTGREEN_EX + '-----------------Ac--------------')
        ac_param = options.pop('activate_param', {'type': 'ReLU'})
        output_map = p_layers.Ac(output_map, ac_param)
        # Convolution
        print(Fore.LIGHTGREEN_EX + '------------------Conv---------------')
        compress_rate = options.pop('compress_rate', None)
        print(Fore.LIGHTGREEN_EX + 'use compress\n compress rate: {0}'.format(compress_rate)) \
            if compress_rate is not None else \
            print(Fore.LIGHTGREEN_EX + 'does not use compress')
        channel = int(output_map.get_shape().as_list()[-1] * (1 - compress_rate)) if compress_rate is not None \
            else output_map.get_shape().as_list()[-1]
        output_map = tf.layers.conv2d(
            output_map,
            filters=channel,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
            kernel_regularizer=layers.l2_regularizer(regularize_weight),
            bias_initializer=tf.zeros_initializer(dtype=param_dtype),
            bias_regularizer=layers.l2_regularizer(regularize_weight),
            name='conv'
        )
        # Optional dropout
        print(Fore.LIGHTGREEN_EX + '------------------dropout---------------')
        dropout_rate = options.pop('dropout_rate', None)
        output_map = tf.layers.dropout(output_map, dropout_rate, training=training, name='dropout') \
            if dropout_rate is not None else output_map
        # Pool
        print(Fore.LIGHTGREEN_EX + '-----------------pool-----------------')
        print('pool kernel: {0}\npool stride: {1}\npool type: {2}'.format(pool_kernel, pool_stride, pool_type))
        if pool_type == 'avg':
            output_map = tf.layers.average_pooling2d(
                output_map,
                pool_kernel,
                pool_stride,
                "same",
                name='pool'
            )
            pass
        elif pool_type == 'max':
            output_map = tf.layers.max_pooling2d(
                output_map,
                pool_kernel,
                pool_stride,
                'same',
                name='pool'
            )
        else:
            print(Fore.LIGHTRED_EX + 'pool method {0} not supported'.format(pool_type))
            sys.exit()
        pass
    return output_map
    pass


def __test_dense_net(config_file):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    output_map = tf.placeholder(tf.float32, shape=[10, 100, 100, 16], name='input')
    training = tf.placeholder(tf.bool, name='training')
    dnp = DenseNetProvide()
    DenseNetFromConfigFile(
        output_map,
        training,
        config_file,
        block_name_flag='dense-',
        dense_net_provide=dnp)
    tf.summary.FileWriter('../test/DenseNet/summary/', tf.Session().graph).close()
    tf.reset_default_graph()
    pass


def __test_de_dense_net(config_file):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    output_map = tf.placeholder(tf.float32, shape=[10, 100, 100, 16], name='input')
    training = tf.placeholder(tf.bool, name='training')
    dnp = DeDenseNetProvide()
    DeDenseNetFromConfigFile(
        output_map,
        training,
        config_file,
        block_name_flag='de_dense-',
        dense_net_provide=dnp)
    tf.summary.FileWriter('../test/DeDenseNet/summary/', tf.Session().graph).close()
    tf.reset_default_graph()
    pass


def __Test(options):
    # todo: test DenseNetBlock
    if options.TestDenseNet:
        config_file = options.DenseNetConfig
        __test_dense_net(config_file)
        pass
    else:
        pass
    if options.TestDeDenseNet:
        config_file = options.DeDenseNetConfig
        __test_de_dense_net(config_file)
    else:
        pass


if __name__ == '__main__':
    from optparse import OptionParser

    usage = "show something usefull -- for example: how to use this program"
    parser = OptionParser(usage)
    parser.add_option(
        '--test_dense_net',
        action='store_true',
        default=False,
        dest='TestDenseNet',
        help='while you want to test the DenseNet set this flag\n'
             'default: False'
    )
    parser.add_option(
        '--dense_net_config',
        action='store',
        default='../test/DenseNet/DenseNet.json',
        dest='DenseNetConfig',
        help='while set test_dense_net , specify the config json file\n'
             'default: ../test/DenseNet/DenseNet.json'
    )
    parser.add_option(
        '--test_de_dense_net',
        action='store_true',
        default=False,
        dest='TestDeDenseNet',
        help='while you want to test the DeDenseNet set this flag\n'
             'default: False'
    )
    parser.add_option(
        '--de_dense_net_config',
        action='store',
        default='../test/DeDenseNet/DeDenseNet.json',
        dest='DeDenseNetConfig',
        help='while set test_de_dense_net , specify the config json file\n'
             'default: ../test/DeDenseNet/DeDenseNet.json'
    )

    (options, args) = parser.parse_args()

    __Test(options)
    pass
