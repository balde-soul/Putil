# coding=utf-8

import Putil.loger as plog
import Putil.tf.static as param
import tensorflow as tf
from colorama import Fore, init, Style
from tensorflow.contrib import layers
import Putil.tf.util as put


root_logger = plog.PutilLogConfig('tf/initialize').logger()
root_logger.setLevel(plog.DEBUG)
InitializeLogger = root_logger.getChild('Initialize')
InitializeLogger.setLevel(plog.DEBUG)
InitializeParamLogger = root_logger.getChild('InitializeParam')
InitializeParamLogger.setLevel(plog.DEBUG)


"""
this class is write to easy fix the param for initialize operation
inherit from param.ParamProbe which provide the method to fix the input param with the default param

this class provide three methods for initialize now , include { 'mrsa', 'zeros', 'xavier'}

usage: 
    at least we should provide 'method' in the param dict:
    param = {'method': 'mrsa'}
    param_fix = InitializeParam(param, default=param['method']).ShowDefault().complement(...).fix_with_default().ParamGenWithInfo()
"""


class InitializeParam(param.ParamProbe):
    def __init__(self, param_feed, **options):
        """

        :param param_feed:
        :param options:
            default: set the default type
        """
        default = options.pop('default', 'mrsa')
        mrsa_param_default = {
            'method': 'mrsa',
            'type': 0.32,
            'initialize_param': {
                'mode': 'fan_avg',
                'scale': 1.0,
                'distribution': 'normal',
                'seed': None
            }
        }
        xavier_param_default = {
            "method": "xavier",
            "type": 0.32,
            "initialize_param": {
                "uniform": True,
                "seed": None
            }
        }
        zero_param_default = {
            'method': 'zeros',
            'type': 0.32,
            "initialize_param": {}
        }

        default_param = {
            'mrsa': mrsa_param_default,
            'zeros': zero_param_default,
            'xavier': xavier_param_default
        }
        super(InitializeParam, self).__init__(default_param[default], param_feed)
        pass

    @staticmethod
    def MRSA():
        return 'mrsa'

    @staticmethod
    def ZEROS():
        return 'zeros'

    @staticmethod
    def XAVIER():
        return 'xavier'

    def ShowDefault(self):
        return super(InitializeParam, self).ShowDefault(logger=InitializeParamLogger)
        pass
    pass


class Initialize:
    def __init__(self, param):
        self._param = param
        pass

    @property
    def Initializer(self):
        return self._initialize()
        pass

    def _initialize(self):
        InitializeLogger.info(Fore.GREEN + 'initializer method: {0}'.format(
            self._param['method']) + Fore.RESET)
        InitializeLogger.info(Fore.GREEN + 'initializer param: {0}'.format(
            param.DictPrintFormat(self._param['initialize_param']) + Fore.RESET))
        if self._param['method'] == InitializeParam.MRSA():
            return self._mrsa_initialize()
        elif self._param['method'] == InitializeParam.ZEROS():
            return self._zero_initialize()
        elif self._param['method'] == InitializeParam.XAVIER():
            return self._xavier_initialize()
        else:
            InitializeLogger.error(
                Fore.RED + 'method: {0} is not supported'.format(self._param['method']) + Fore.RESET)
            raise ValueError()
        pass

    def _mrsa_initialize(self):
        init = tf.variance_scaling_initializer(
            scale=self._param['initialize_param']['scale'],
            mode=self._param['initialize_param']['mode'],
            distribution=self._param['initialize_param']['distribution'],
            seed=self._param['initialize_param']['seed'],
            dtype=put.tf_type(self._param['type']).Type
        )
        return init
        pass

    def _zero_initialize(self):
        init = tf.zeros_initializer(
            dtype=put.tf_type(self._param['type']).Type
        )
        return init
        pass

    def _xavier_initialize(self):
        init = layers.xavier_initializer(
            uniform=self._param['initialize_param']['uniform'],
            seed=self._param['initialize_param']['seed'],
            dtype=put.tf_type(self._param['type']).Type
        )
        return init
        pass
    pass


if __name__ == '__main__':

    pass
