# coding=utf-8

import Putil.loger as plog
import Putil.tf.static as param
import tensorflow as tf
from colorama import Fore, init, Style
from tensorflow.contrib import layers


root_logger = plog.PutilLogConfig('tf/param/ac').logger()
root_logger.setLevel(plog.DEBUG)
ACParamLogger = root_logger.getChild('ACParam')
ACParamLogger.setLevel(plog.DEBUG)
ACLogger = root_logger.getChild('AC')
ACLogger.setLevel(plog.DEBUG)


class ACParam(param.ParamProbe):
    def __init__(self, param_feed, **options):
        self.default = options.pop('default', 'PReLU')
        prelu_default = {
            'method': 'PReLU',
            'type': 0.32,
            'ac_param': {
                'name': '',
                'alpha': 0.1,
                'trainable': True
            }
        }
        default_param = {
            'PReLU': prelu_default
        }
        param.ParamProbe.__init__(self, default_param[self.default], param_feed)
        pass

    @staticmethod
    def PRELU():
        return 'PReLU'
        pass
    pass


class AC:
    def __init__(self, param, input):
        self._param = param
        self._input = input

        #PReLU
        self._alpha = None
        self._pos = None
        self._neg = None
        self._prelu = None
        pass

    @property
    def PReLU_Alpha(self):
        return self._alpha

    @property
    def PReLU_Pos(self):
        return self._pos

    @property
    def PReLU_Neg(self):
        return self._neg

    @property
    def PReLU_PReLU(self):
        return self._prelu

    @property
    def LayerOutput(self):
        if self._param['method'] == ACParam.PRELU():
            return self.__prelu()
            pass
        else:
            ACLogger.error(Fore.RED +
                           'method: {0} is not support'.format(self._param['method'])
                           + Fore.RESET
                           )
            raise ValueError()
        pass

    def __prelu(self):
        if self._param['ac_param']['trainable']:
            with tf.variable_scope(self._param['ac_param']['name']):
                self._alpha = tf.Variable(
                    self._param['ac_param']['alpha'],
                    name='{0}-alpha'.format(self._param['ac_param']['name']))
                self._pos = tf.nn.relu(self._input)
                self._neg = self._alpha * (self._input - tf.abs(self._input)) * 0.5
                self._prelu = self._pos + self._neg
                pass
            return self._prelu
        else:
            with tf.variable_scope(self._param['ac_param']['name']):
                self._prelu = tf.nn.leaky_relu(self._input, self._param['ac_param']['alpha'])
                pass
        pass

