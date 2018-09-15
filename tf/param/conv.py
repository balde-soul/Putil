# coding=utf-8

import Putil.loger as plog
import Putil.tf.static as param
import Putil.tf.param.initialize as init
import tensorflow as tf
import Putil.tf.util  as tfu
from colorama import Fore, init, Style
from tensorflow.contrib import layers


root_logger = plog.PutilLogConfig('tf/conv').logger()
root_logger.setLevel(plog.DEBUG)
ConvLogger = root_logger.getChild('Conv')
ConvLogger.setLevel(plog.DEBUG)
ConvParamLogger = root_logger.getChild('ConvParam')
ConvParamLogger.setLevel(plog.DEBUG)


"""
convolution layer param:
{
    method: **,
    type: **,
    conv_param: {
        filter: **,
        kernel: **,
        stride: **,
        padding: **,
        dilation: [**, **]
    },
    bias_initialize_method: {
        InitializeParam
    },
    bias_initialize_method: {
        InitializeParam
    }
}
"""


class ConvParam(param.ParamProbe):
    def __init__(self, param_feed, **options):
        """

        :param param_feed:
        :param options:
        """
        self.default = options.pop('default', 'two_with_bias')
        self._bias_initializer_param_feed = \
            param_feed.get('bias_initialize_method', {})
        self._weight_initializer_param_feed = \
            param_feed.get('bias_initialize_method', {})
        self._bias_initializer = init.InitializeParam(
            self._bias_initializer_param_feed,
            default=self._bias_initializer_param_feed.get('method', 'mrsa')
        )
        self._weight_initializer = init.InitializeParam(
            self._weight_initializer_param_feed,
            default=self._weight_initializer_param_feed.get('method', 'mrsa')
        )
        two_with_bias_default = {
            "method": "2d_with_bias",
            'type': 0.32,
            'conv_param': {
                'name': '',
                'filter': 10,
                'kernel': [3, 3],
                'stride': [1, 1],
                'padding': 'same',
                'dilation': [1, 1],
                'gpu': True,
            }
        }
        default_param = {
            'two_with_bias': two_with_bias_default
        }
        param.ParamProbe.__init__(self, default_param[self.default], param_feed)
        pass

    @staticmethod
    def TWO_WITH_BIAS():
        ConvParamLogger.debug(Fore.GREEN +
                              'choose two_with_bias'
                              + Fore.RESET)
        return 'two_with_bias'

    def fix_with_default(self):
        self._weight_initializer.fix_with_default()
        self._bias_initializer.fix_with_default()
        param.ParamProbe.fix_with_default(self)
        return self
        pass

    def bias_complement(self, **options):
        self._bias_initializer.complement(**options)
        return self
        pass

    def weight_complement(self, **options):
        self._weight_initializer.complement(**options)
        return self
        pass


class Conv:
    def __init__(self, param, input):
        self._input = input
        self._param = param

        self._weight = None
        self._bias = None
        self._conv = None
        self._bias_add = None
        pass

    @property
    def Weiht(self):
        return self._weight

    @property
    def Bias(self):
        return self._bias

    @property
    def Conv(self):
        return self._conv

    @property
    def BiasAdd(self):
        return self._bias_add

    @property
    def LayerOutput(self):
        ConvLogger.info(Fore.GREEN +
                        'conv method: {0}'.format(
                            self._param['method']
                        )
                        + Fore.RESET)
        ConvLogger.info(Fore.GREEN +
                        'conv param: {0}'.format(
                            param.DictPrintFormat(self._param['conv_param'])
                        )
                        + Fore.RESET)
        if self._param['method'] == ConvParam.TWO_WITH_BIAS():
            return self.__two_with_bias()
        else:
            ConvLogger.error(Fore.RED +
                             'method: {0} is not supported'.format(self._param['method'])
                             + Fore.RESET)
            raise ValueError()
        pass

    def __two_with_bias(self):
        self._weight = tf.get_variable(
            name='{0}-conv_weight'.format(self._param['conv_param']['name']),
            shape=self._param['conv_param']['kernel'] +
                  [self._input.get_shape().as_list()[-1]] +
                  [self._param['conv_param']['filter']],
            dtype=tfu.tf_type(self._param['type']).Type,
            initializer=init.Initialize(self._param['weight_initialize_param']).Initializer
        )
        self._bias = tf.get_variable(
            name='{0}-conv_weight'.format(self._param['conv_param']['name']),
            shape=[self._param['conv_param']['filter']],
            dtype=tfu.tf_type(self._param['type']).Type,
            initializer=init.Initialize(self._param['bias_initialize_param']).Initializer
        )
        self._conv = tf.nn.conv2d(
            self._input,
            self._weight,
            [1] + self._param['conv_param']['stride'] + [1],
            self._param['conv_param']['padding'],
            self._param['conv_param']['gpu'],
            dilations=[1] + self._param['conv_param']['dilation'] + [1],
            name=self._param['conv_param']['name'] + '_conv'
        )
        self._bias_add = tf.nn.bias_add(
            self._conv,
            self._bias,
            name='{0}'.format(self._param['conv_param']['name'])
        )
        return self._bias_add
        pass
